# from fastNLP.core import batch
from heapq import merge
import sys
from uu import Error
from parsing_by_maxseminfo.utils.utils import compute_oneside_matching_with_spanoverlap_score, compute_oneside_matching_with_spanoverlap_score_competing, compute_oneside_matching_with_spanoverlap_score_merged
import torch
import torch.nn as nn
from parsing_by_maxseminfo.parser.modules.res import ResLayer, ResLayerWithNorm
from ..pcfgs.pcfg import PCFG, FasterMyPCFG, MyPCFG


class NeuralPCFG(nn.Module):
    def __init__(self, args, dataset):
        super(NeuralPCFG, self).__init__()
        self.pcfg = PCFG()
        self.device = dataset.device
        self.args = args

        self.NT = args.NT
        self.T = args.T
        self.V = len(dataset.word_vocab)

        self.s_dim = args.s_dim

        self._para_init()
        self._initialize()

    def _para_init(self):

        self.term_emb = nn.Parameter(torch.randn(self.T, self.s_dim))
        self.nonterm_emb = nn.Parameter(torch.randn(self.NT, self.s_dim))
        self.root_emb = nn.Parameter(torch.randn(1, self.s_dim))

        self.term_mlp = nn.Sequential(
            nn.Linear(self.s_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
            nn.Linear(self.s_dim, self.V),
        )


        self.root_mlp = nn.Sequential(
            nn.Linear(self.s_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
            nn.Linear(self.s_dim, self.NT),
        )



        self.NT_T = self.NT + self.T
        self.rule_mlp = nn.Linear(self.s_dim, (self.NT_T) ** 2)
        

    def _initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, input, evaluating=False):
        x = input["word"]
        device = x.device
        b, n = x.shape[:2]

        def roots():
            root_emb = self.root_emb
            roots = self.root_mlp(root_emb)
            # if self.args.use_bn:
                # roots = self.root_bn(roots)
            roots = roots.log_softmax(-1)
            return roots.expand(b, self.NT)

        def terms():
            assert not torch.isnan(self.term_emb).any(), f"term_emb contains nan, {torch.isnan(self.term_emb).sum()}"
            assert not torch.isinf(self.term_mlp(self.term_emb)).any(), "term mlp contains inf"
            assert not torch.isnan(self.term_mlp(self.term_emb)).any(), "term mlp output has nan "
            term_projection = self.term_mlp(self.term_emb)
            # if self.args.use_bn:
                # term_projection = self.term_bn(term_projection)
            term_prob = term_projection.log_softmax(-1)
            assert not torch.isnan(term_prob).any(), "term log prob has nan"
            return term_prob[torch.arange(self.T)[None, None], x.cpu()[:, :, None]]

        def rules():
            rule_prob = self.rule_mlp(self.nonterm_emb)
            # if self.args.use_bn:
                # rule_prob = self.rule_bn(rule_prob)
            rule_prob = rule_prob.log_softmax(-1)
            rule_prob = rule_prob.reshape(self.NT, self.NT_T, self.NT_T)
            return rule_prob.unsqueeze(0).expand(b, *rule_prob.shape).contiguous()

        root, unary, rule = roots(), terms(), rules()

        return {
            "unary": unary,
            "root": root,
            "rule": rule,
            "kl": torch.tensor(0, device=device),
        }

    def loss(self, input):
        rules = self.forward(input)
        result = self.pcfg._inside(rules=rules, lens=input["seq_len"])
        return -result["partition"].mean()

    def evaluate(self, input, decode_type, **kwargs):
        rules = self.forward(input, evaluating=True)
        if decode_type == "viterbi":
            return self.pcfg.decode(
                rules=rules, lens=input["seq_len"], viterbi=True, mbr=False
            )
        elif decode_type == "mbr":
            return self.pcfg.decode(
                rules=rules, lens=input["seq_len"], viterbi=False, mbr=True
            )
        else:
            raise NotImplementedError


class NeuralPCFGPairwise(NeuralPCFG):
    def __init__(self, args, vocab_size):
        super(NeuralPCFG, self).__init__()
        self.pcfg = MyPCFG()
        # self.device = device
        self.args = args

        self.NT = args.NT
        self.T = args.T
        self.V = vocab_size

        self.s_dim = args.s_dim

        self._para_init()
        self._initialize()


    def loss(self, input_target, input_pas):
        raise NotImplementedError("This class serve only as a base class")

    def evaluate(
        self, input_target, input_pas=None, decode_type="mbr", eval_wd=False, span_mask = None,  **kwargs
    ):
        if decode_type == "mbr":
            # assert decode_type == "mbr", "currently only the mbr decoding mode is supported"
            marginal_device = input_target["word"].device
            rules_target = self.forward(input_target)
            target_pack = self.pcfg.decode(
                rules=rules_target,
                lens=input_target["seq_len"],
                viterbi=False,
                mbr=True,
                allow_grad=False,
                span_mask=span_mask,
            )
            if input_pas is None:
                return target_pack
            target_span_marginals = target_pack["span_marginals"]
            target_span_marginals = target_span_marginals / (input_target["seq_len"] - 1)[
                :, None, None
            ].to(marginal_device)
            target_predictions = target_pack["prediction"]
            target_partitions = target_pack["partition"]

            rules_pas = self.forward(input_pas, evaluating=True)
            pas_span_marginals = self.pcfg.decode(
                rules=rules_pas,
                lens=input_pas["seq_len"],
                # span_dist=True,
                mbr=True,
                allow_grad=False,
            )["span_marginals"]
            pas_span_marginals = pas_span_marginals / (input_pas["seq_len"] - 1)[
                :, None, None
            ].to(marginal_device)

            return {
                "partition": target_partitions,
                "prediction": target_predictions,
                "span_marginals": target_span_marginals,
                "pas_span_marginals": pas_span_marginals,
            }
        else:
            marginal_device = input_target["word"].device
            rules_target = self.forward(input_target)
            target_pack = self.pcfg.decode(
                rules=rules_target,
                lens=input_target["seq_len"],
                viterbi=False,
                mbr=True,
                allow_grad=False,
                span_mask=span_mask,
            )
            return target_pack




class NeuralPCFGOT(NeuralPCFGPairwise):

    def __init__(self, args, vocab_size, span_repr_mode, langstr, **kwargs):
        super().__init__(args, vocab_size)
        self.wvmodel = None
        if hasattr(self.args, "use_fast_pcfg" ) and  self.args.use_fast_pcfg:
            self.pcfg = FasterMyPCFG()

    pass

    @torch.enable_grad()
    def loss(
        self,
        input_target,
        input_pas,
        bge_model,
        pasid2groupid_map,
        span_mask=None,
        run_wd=True,
        max_bandwidth = 1000,
        gumbel_softmax_tau = 1.0,
        flag_use_ppl_as_nll_loss = False,
        flag_span_repr_mode = "bge-m3",
        flag_similarity_mode = "inner_product",
    ):
      raise NotImplementedError("This class serves only as a base class")
    

from parser.model.N_PCFG import NeuralPCFGPairwise, NeuralPCFGOT
class NeuralPCFGFixedCostReward(NeuralPCFGOT):
    @torch.enable_grad()
    def loss(
        self,
        input_pack,
        run_wd=True,
        dropout = 0.,
        mode = None,
        rl_coeff=1. ,
        maxent_coeff = 1.,
        sample_epsilon = 0.,
        include_unary = False,
        num_samples = 4,
        sample_mode = "crf",
        supervised_mode = False,
    ):
        assert sample_mode == "crf", "This class support only the CRF mode"
        batch_size, maxlen = input_pack["word"].shape
        marginal_device = input_pack["word"].device
        rules_target = self.forward(input_pack)

        span_dist= True
        run_wd = True
        result_target = self.pcfg._inside(
            rules=rules_target,
            lens=input_pack["seq_len"],
            span_dist=span_dist,
            allow_grad=run_wd,
            span_mask=None,
            include_unary=include_unary
        )
        if supervised_mode:
            raise NotImplementedError("Please use the A2C class for the supervised experiment")
            


        rewards, greedy_rewards, ent, _ = self.pcfg.sample_crf(grad=result_target["span_marginals"].sum(-1), reward=input_pack["reward"], num_samples=num_samples, epsilon=sample_epsilon)

        rv = torch.tensor([[s[1] for s in b] for b in rewards])
        logp = torch.stack([torch.stack([s[0] for s in b], dim=0) for b in rewards], dim=0)
        rv_greedy = torch.tensor([[s[1] for s in b] for b in greedy_rewards])
        rvbl = rv.mean(1, keepdim=True)
        rw = (rv - rvbl)
        loss_pg = torch.mean(-logp*(rw + maxent_coeff*ent.unsqueeze(-1).cpu()), dim=-1) # policy gradient loss, per instance
        assert loss_pg.shape[0] == batch_size



        stat_rv = rv.mean(-1).mean(-1)
        stat_best_rv = rv_greedy.squeeze(-1).mean(-1)
        stat_nll = -result_target["partition"].mean()
        stat_logppl = stat_nll/maxlen
        stat_entropy = ent.mean(-1)
        stat_rw = rw.std(-1).mean(-1)
        


        c_nll = mode == "nll"
        c_rlonly = mode == "rlonly"
        c_rl = mode == "rl"
        sum([c_nll, c_rlonly, c_rl]) == 1, "only one mode can be selected"
        if c_nll:
            loss = -result_target["partition"]
        elif c_rlonly:
            loss = rl_coeff * loss_pg
        elif c_rl:
            # loss_pg/=(maxlen) # logp scaling () normalize by sequence length
            loss = rl_coeff * loss_pg + -result_target["partition"].cpu()
        else:
            raise NotImplementedError(f"{mode} not implemented")



        return {
            "loss": loss.mean(),#spannll_loss + spancomp_loss_weight * spancomp_loss,
            "stat_nll": stat_nll,
            "stat_reward": stat_rv,
            "stat_baseline_reward": stat_best_rv,
            "stat_logppl": stat_logppl,
            "stat_entropy": stat_entropy,
            "stat_rw": stat_rw,
            "coeff_rl": rl_coeff if not c_nll else 0,
            "coeff_maxent": maxent_coeff if not c_nll else 0,
            # "flag_rl_len_norm": rl_len_norm,
            # "span_marginals": target_span_marginals,
            # "tocopy_array": result_target["tocopy_array"],
        }



    def train(self, mode=True):
        super().train(mode)
        self.pcfg.train(mode)


    
class NeuralPCFGFixedCostRewardA2C(NeuralPCFGFixedCostReward): 
    @torch.enable_grad()
    def loss(
        self,
        input_pack,
        run_wd=True,
        dropout=0.0,
        mode=None,
        rl_coeff=1. ,
        maxent_coeff = 1.,
        sample_epsilon = 0.,
        include_unary = False,
        num_samples = 4,
        supervised_mode = False,
        sample_mode = "crf",
        tau = 1.,


    ):
        # assert sample_mode == "pcfg", "tempoary trigger to force PCFG mode for SNPCFG"
        assert sample_mode in ["pcfg", "crf"]
        batch_size, maxlen = input_pack["word"].shape
        marginal_device = input_pack["word"].device
        rules_target = self.forward(input_pack)

        if sample_mode == "pcfg":
            span_dist = False
            run_wd = False
        else:
            span_dist = True
            run_wd = True

        result_target = self.pcfg._inside(
            rules=rules_target,
            lens=input_pack["seq_len"],
            span_dist=span_dist,
            allow_grad=run_wd,
            span_mask=None,
            dropout=dropout,
            include_unary=include_unary,
            reward=input_pack["reward"] #if sample_mode=="pcfg" else None
        )

        if supervised_mode:
            if sample_mode == "crf":
                span_marginals = result_target["span_marginals"] 
                gold_tree_mask = input_pack["gold_tree_mask"]
                loss = -(torch.clamp(span_marginals.sum(-1), min=1e-20, max=1.)).log()
                hit_count = gold_tree_mask.flatten(start_dim=1).sum(-1)
                loss = (loss*gold_tree_mask).flatten(start_dim=1).sum(-1)/hit_count - result_target["partition"]
                stat_logppl = -result_target["partition"].mean()/maxlen
                return {
                    "loss": loss.mean(),
                    "stat_nll": -result_target["partition"].mean(),
                    "stat_reward": torch.tensor(0),
                    "stat_baseline_reward": torch.tensor(0),
                    "stat_logppl": stat_logppl,#torch.tensor(0),
                    "stat_entropy": torch.tensor(0),
                    "stat_rw": torch.tensor(0),
                    "coeff_rl": torch.tensor(0),
                    "coeff_maxent": torch.tensor(0),
                    "flag_rl_len_norm": torch.tensor(0),
                }
            elif sample_mode == "pcfg":
                gold_tree_mask = input_pack["tree_compatible_masks"]
                result_treeonly = self.pcfg._inside(
                    rules=rules_target,
                    lens=input_pack["seq_len"],
                    span_dist=False,
                    allow_grad=False,
                    span_mask=gold_tree_mask,
                    dropout=dropout,
                    include_unary=include_unary,
                    reward=input_pack["reward"]
                )
                loss = -result_treeonly["partition"] #+ result_target["partition"]
                stat_logppl = -result_target["partition"].mean()/maxlen
                return {
                    "loss": loss.mean(),
                    "stat_nll": -result_target["partition"].mean(),
                    "stat_reward": torch.tensor(0),
                    "stat_baseline_reward": torch.tensor(0),
                    "stat_logppl": stat_logppl,#torch.tensor(0),
                    "stat_entropy": torch.tensor(0),
                    "stat_rw": torch.tensor(0),
                    "coeff_rl": torch.tensor(0),
                    "coeff_maxent": torch.tensor(0),
                    "flag_rl_len_norm": torch.tensor(0),
                }

        if sample_mode == "crf" or sample_mode == "a2c":
            rewards, greedy_rewards, ent, v_est = self.pcfg.sample_crf(grad=result_target["span_marginals"].sum(-1), reward=input_pack["reward"], num_samples=num_samples, epsilon=sample_epsilon, span_score=[1])
        elif sample_mode == "pcfg":
            root = rules_target["root"]

            rewards, greedy_rewards, ent = self.pcfg.sample_pcfg_entrance(prod = (None, None, None, rules_target), beta=(result_target["beta"], rules_target["unary"]), reward=input_pack["reward"], num_samples=num_samples, epsilon=sample_epsilon, root_rule=root, tau=tau, span_score= result_target["v_est"])
            v_est = result_target["v_est"]

        else: raise NotImplementedError

        assert torch.all(ent >= 0), "entropy should be non-negative, got {}".format(ent)

        rv = torch.tensor([[s[1] for s in b] for b in rewards])
        if sample_mode == "crf":
            logp = torch.stack([torch.stack([s[0] for s in b], dim=0) for b in rewards], dim=0).cpu()
            logp_ss = torch.stack([torch.stack([s[4] for s in b], dim=0) for b in rewards], dim=0).cpu()
            adv_ss = torch.stack([torch.stack([s[5] for s in b], dim=0) for b in rewards], dim=0).cpu()
        elif sample_mode == "pcfg":
            logp = torch.stack([torch.stack([s[0]  for s in b], dim=0) for bid, b in enumerate(rewards)], dim=0).cpu()
        else: raise NotImplementedError
        

        c_nll = mode == "nll"
        c_rlonly = mode == "rlonly"
        c_rl = mode == "rl"
        c_a2c = mode == "a2c"
        c_ta2c = mode == "ta2c"
        c_ta2c_rules = mode == "ta2c_rules"
        c_a2c_v0 = mode == "a2c_v0"
        c_tavg = mode == "tavg"
        assert sum([c_nll, c_rlonly, c_rl, c_a2c, c_ta2c, c_ta2c_rules, c_a2c_v0, c_tavg]) == 1, "only one mode can be selected"
        rw=torch.tensor([[0.]])
        stat_a2c_loss = torch.tensor([0.])
        
        
        if c_nll:
            loss = -result_target["partition"]

        elif c_rlonly:
            rv = torch.tensor([[s[1] for s in b] for b in rewards])
            rvbl = rv.mean(1, keepdim=True)
            rw = (rv - rvbl)
            logp = torch.stack([torch.stack([s[0]  for s in b], dim=0) for bid, b in enumerate(rewards)], dim=0).cpu()
            loss_pg = torch.mean(-logp*(rw + maxent_coeff*ent.unsqueeze(-1).cpu()), dim=-1) # policy gradient loss, per instance
            loss = rl_coeff * loss_pg
        elif c_rl:
            rv = torch.tensor([[s[1] for s in b] for b in rewards])
            rvbl = rv.mean(1, keepdim=True)
            rw = (rv - rvbl)
            logp = torch.stack([torch.stack([s[0]  for s in b], dim=0) for bid, b in enumerate(rewards)], dim=0).cpu()
            loss_pg = torch.mean(-logp*(rw + maxent_coeff*ent.unsqueeze(-1).cpu()), dim=-1) # policy gradient loss, per instance
            loss = rl_coeff * loss_pg + -result_target["partition"].cpu()
        elif c_a2c:
            logp_ss = torch.stack([torch.stack([s[4] for s in b], dim=0) for b in rewards], dim=0).cpu()
            adv_ss = torch.stack([torch.stack([s[5] for s in b], dim=0) for b in rewards], dim=0).cpu().detach()
            ent_ss = torch.stack([torch.stack([s[6] for s in b], dim=0) for b in rewards], dim=0).cpu()
            loss_a2c = (-logp_ss*(adv_ss + maxent_coeff * ent_ss)).sum(-1).mean(-1) # policy gradient loss, per instance
            if sample_mode == "pcfg":
                loss_a2c = (-logp_ss*(adv_ss + maxent_coeff * ent_ss)).mean(-1).mean(-1) # policy gradient loss, per instance
            loss = rl_coeff * loss_a2c.cpu() + -result_target["partition"].cpu()
            stat_a2c_loss = loss_a2c.mean(-1)
        elif c_a2c_v0:
            assert sample_mode == "crf", "sample mode must be crf"
            logp = torch.stack([torch.stack([s[0] for s in b], dim=0) for b in rewards], dim=0).cpu()
            v0 = v_est[:, 0, v_est.shape[1]-1].unsqueeze(-1)
            rv = torch.tensor([[s[1] for s in b] for b in rewards])
            rw = (rv.cpu() - v0.cpu()).detach()
            loss_a2c = (-logp*(rw + maxent_coeff*ent.unsqueeze(-1).cpu())).mean(-1) # policy gradient loss, per instance
            loss = rl_coeff * loss_a2c - result_target["partition"].cpu()
            stat_a2c_loss = loss_a2c.mean(-1)
        elif c_ta2c:
            # Note here we CANNOT normalize the tree logp against the sentence logp, since it will result in the unlearning of the grammar.
            rv = torch.tensor([[s[1] for s in b] for b in rewards])
            logp = torch.stack([torch.stack([s[3]  for s in b], dim=0) for bid, b in enumerate(rewards)], dim=0).cpu()
            v0 = result_target["v0"]
            rw = (rv.cpu() - v0.unsqueeze(-1).cpu())
            rw_normalized = rw.detach()

            loss_a2c = (-logp.cpu() * rw_normalized).mean(-1)
            loss = rl_coeff * loss_a2c.cpu() + -result_target["partition"].cpu()
            stat_a2c_loss = loss_a2c.mean(-1)
        elif c_ta2c_rules:
            logp_ss = torch.stack([torch.stack([s[4] for s in b], dim=0) for b in rewards], dim=0).cpu()
            adv_ss = torch.stack([torch.stack([s[5] for s in b], dim=0) for b in rewards], dim=0).cpu().detach()
            ent_ss = torch.stack([torch.stack([s[6] for s in b], dim=0) for b in rewards], dim=0).cpu()
            loss_a2c = (-logp_ss*(adv_ss + maxent_coeff * ent_ss)).sum(-1).mean(-1) # policy gradient loss, per instance
            loss = rl_coeff * loss_a2c.cpu() + -result_target["partition"].cpu()
            stat_a2c_loss = loss_a2c.mean(-1)
        elif c_tavg:
            assert sample_mode == "pcfg", "sample mode must be crf"
            logp = torch.stack([torch.stack([s[0] for s in b], dim=0) for b in rewards], dim=0).cpu()
            rv = torch.tensor([[s[1] for s in b] for b in rewards])
            rw = (rv - rv.mean(-1).unsqueeze(-1)).cpu().detach()
            loss_a2c = (-logp*(rw + maxent_coeff*ent.unsqueeze(-1).cpu())).mean(-1) # policy gradient loss, per instance
            loss = rl_coeff * loss_a2c - result_target["partition"].cpu()
            stat_a2c_loss = loss_a2c.mean(-1)


        else:
            raise NotImplementedError(f"{mode} not implemented")


        stat_rv = rv.mean(-1).mean(-1)
        rv_greedy = torch.tensor([[s[1] for s in b] for b in greedy_rewards])
        stat_best_rv = rv_greedy.squeeze(-1).mean(-1)
        stat_nll = -result_target["partition"].mean()
        stat_logppl = stat_nll/maxlen
        stat_entropy = ent.mean(-1)
        stat_rw = rw.std(-1).mean(-1) #if not isinstance(rw, int) else 0
        stat_logp = logp.mean(-1)
        stat_vest = v_est
        return {
            "loss": loss.mean(),#spannll_loss + spancomp_loss_weight * spancomp_loss,
            "stat_nll": stat_nll,
            "stat_reward": stat_rv,
            "stat_baseline_reward": stat_best_rv,
            "stat_logppl": stat_logppl,
            "stat_entropy": stat_entropy,
            "stat_rw": stat_rw,
            "stat_logp": stat_logp,
            "stat_vest": stat_vest,
            "stat_a2c_loss": stat_a2c_loss,
            "coeff_rl": rl_coeff if not c_nll else 0,
            "coeff_maxent": maxent_coeff if not c_nll else 0,
            # "flag_rl_len_norm": rl_len_norm,
        }
    
    def train(self, mode=True):
        super().train(mode)
        self.pcfg.train(mode)
