import torch
import torch.nn as nn
from parsing_by_maxseminfo.parser.modules.res import ResLayer, ResLayerWithNorm
from parsing_by_maxseminfo.parser.pcfgs.simple_pcfg import MyPCFG_Triton, MySPCFGFaster, SimplePCFG_Triton


class Simple_N_PCFG(nn.Module):
    def __init__(self, args, dataset):
        super(Simple_N_PCFG, self).__init__()
        self.pcfg = SimplePCFG_Triton()
        self.device = dataset.device
        self.args = args
        self.NT = args.NT
        self.T = args.T
        self.V = (
            len(dataset.word_vocab)
            if hasattr(dataset, "word_vocab")
            else len(dataset.V)
        )
        self.s_dim = args.s_dim
        self.rule_dim = self.s_dim
        self._para_init()
        self._initialize()

    def _para_init(self):
        ## root
        self.root_emb = nn.Parameter(torch.randn(1, self.s_dim))

        # terms
        self.term_mlp = nn.Sequential(
            nn.Linear(self.s_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
        )

        self.vocab_emb = nn.Parameter(torch.randn(self.s_dim, self.V))

        self.rule_state_emb = nn.Parameter(torch.randn(self.NT + self.T, self.s_dim))

        self.left_mlp = nn.Sequential(
            nn.Linear(self.rule_dim, self.rule_dim), nn.ReLU()
        )
        self.right_mlp = nn.Sequential(
            nn.Linear(self.rule_dim, self.rule_dim), nn.ReLU()
        )
        self.parent_mlp1 = nn.Sequential(
            nn.Linear(self.s_dim, self.s_dim),
            nn.ReLU(),
        )

    def _initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, input, **kwargs):

        x = input["word"]
        b, n = x.shape[:2]

        def roots():
            roots = self.root_emb @ self.rule_state_emb[: self.NT].t()
            roots = roots.log_softmax(-1)
            return roots.expand(b, roots.shape[-1])

        def terms():
            term_emb = self.rule_state_emb[self.NT :]
            term_prob = (
                (self.term_mlp(term_emb) + term_emb) @ self.vocab_emb
                # (self.term_mlp(term_emb) ) @ self.vocab_emb
            ).log_softmax(-1)
            return term_prob[torch.arange(self.T)[None, None], x[:, :, None]]

        def rules():
            rule_state_emb = self.rule_state_emb
            nonterm_emb = rule_state_emb[: self.NT]
            parent1 = self.parent_mlp1(nonterm_emb) + nonterm_emb

            # parent2 = self.parent_mlp2(nonterm_emb) + nonterm_emb
            assert torch.all(~torch.isnan(rule_state_emb))
            # assert torch.all(~torch.isnan(rule_state_emb))
            assert torch.all(~torch.isnan(parent1))


            left = (self.left_mlp(rule_state_emb) + rule_state_emb) @ parent1.t()
            right = (self.right_mlp(rule_state_emb) + rule_state_emb) @ parent1.t()
            # right = left
            assert torch.all(~torch.isnan(left))
            assert torch.all(~torch.isnan(right))

            # head = head.softmax(-1)
            left = left.softmax(-2)
            right = right.softmax(-2)

            left_m = left[: self.NT, :]
            left_p = left[self.NT :, :]

            right_m = right[: self.NT, :]
            right_p = right[self.NT :, :]

            return (left_m, left_p, right_m, right_p)

        root, unary, (left_m, left_p, right_m, right_p) = roots(), terms(), rules()

        return {
            "unary": unary,
            "root": root,
            "left_m": left_m,
            "right_m": right_m,
            "left_p": left_p,
            "right_p": right_p,
            "kl": 0,
        }

    def loss(self, input):
        rules = self.forward(input)
        result = self.pcfg._inside(rules=rules, lens=input["seq_len"])
        logZ = -result["partition"].mean()
        return logZ

    def evaluate(self, input, decode_type, **kwargs):
        rules = self.forward(input)
        if decode_type == "viterbi":
            assert NotImplementedError

        elif decode_type == "mbr":
            return self.pcfg.decode(
                rules=rules, lens=input["seq_len"], viterbi=False, mbr=True
            )
        else:
            raise NotImplementedError


class SNPCFGPairwise(Simple_N_PCFG):
    def __init__(self, args, vocab_size):
        super(Simple_N_PCFG, self).__init__()
        self.pcfg = MySPCFGFaster()
        # self.device = device
        self.args = args

        self.NT = args.NT
        self.T = args.T
        self.V = vocab_size

        self.s_dim = args.s_dim
        self.rule_dim = self.s_dim

        self._para_init()
        self._initialize()

    def evaluate(
        self,
        input_target,
        input_pas=None,
        decode_type="mbr",
        eval_wd=False,
        span_mask=None,
        **kwargs,
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



class SNPCFGOT(SNPCFGPairwise):

    def __init__(self, args, vocab_size, span_repr_mode, langstr):
        super().__init__(args, vocab_size)
        self.wvmodel = None

    pass

class SNPCFGFixedCostReward(SNPCFGOT):
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


    ):

        assert sample_mode == "crf", "This class support only the CRF mode"
        batch_size, maxlen = input_pack["word"].shape

        rules_target = self.forward(input_pack)


        span_dist = True
        run_wd = True

       
        result_target = self.pcfg._inside(
            rules=rules_target,
            lens=input_pack["seq_len"],
            span_dist=span_dist,
            allow_grad=run_wd,
            span_mask=None,
            dropout=dropout,
            include_unary=include_unary
        )

        if supervised_mode:
            raise NotImplementedError("Please use the A2C class for the supervised experiment")



        rewards, greedy_rewards, ent, _ = self.pcfg.sample_crf(grad=result_target["span_marginals"].sum(-1), reward=input_pack["reward"], num_samples=num_samples, epsilon=sample_epsilon)
        
        rv = torch.tensor([[s[1] for s in b] for b in rewards])
        logp = torch.stack([torch.stack([s[0] for s in b], dim=0) for b in rewards], dim=0).cpu()
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
        stat_logp = logp.mean(-1)
        # stat_rb_reward = rb_rewards.mean(-1)
        

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
            "stat_logp": stat_logp,
            "coeff_rl": rl_coeff if not c_nll else 0,
            "coeff_maxent": maxent_coeff if not c_nll else 0,
            # "flag_rl_len_norm": rl_len_norm,
        }

    def train(self, mode=True):
        super().train(mode)
        self.pcfg.train(mode)


#     def train(self, mode=True):
#         super().train(mode)
#         self.pcfg.train(mode)

class SNPCFGFixedCostRewardA2C(SNPCFGFixedCostReward): 
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
        assert sample_mode in ["pcfg", "crf"]




        # print(result_target["span_marginals"].sum(-1).shape)
        # print(x_target["reward"].shape)
        batch_size, maxlen = input_pack["word"].shape


        marginal_device = input_pack["word"].device
        rules_target = self.forward(input_pack)
        # print("rules_target", rules_target["root"].shape, rules_target["unary"].shape, rules_target["left_m"].shape, rules_target["right_m"].shape, rules_target["left_p"].shape, rules_target["right_p"].shape)
        # input()

        if sample_mode == "pcfg":
            span_dist = False
            run_wd = False
        else:
            span_dist = True
            run_wd = True

        assert len(rules_target["left_m"].shape) == 2, f"left_m shape should be (NT, NT), {rules_target['right_m'].shape}"
        assert len(rules_target["right_m"].shape) == 2, f"right_m shape should be (NT, NT), {rules_target['right_m'].shape}" 


        result_target = self.pcfg._inside(
            rules=rules_target,
            lens=input_pack["seq_len"],
            span_dist=span_dist,
            allow_grad=run_wd,
            span_mask=None,
            dropout=dropout,
            include_unary=include_unary,
            reward=input_pack["reward"] if not supervised_mode else None
        )

        if supervised_mode:
            if sample_mode == "crf":
                span_marginals = result_target["span_marginals"] 
                gold_tree_mask = input_pack["gold_tree_mask"]
                loss = -(torch.clamp(span_marginals.sum(-1), min=1e-20, max=1.)).log()
                assert loss.shape == gold_tree_mask.shape
                hit_count = gold_tree_mask.flatten(start_dim=1).sum(-1)
                loss = (loss*gold_tree_mask).flatten(start_dim=1).sum(-1)/hit_count - result_target["partition"]
                assert loss.shape == hit_count.shape
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
                    reward=None,#input_pack["reward"]
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



        # result_target['beta'] = torch.where(result_target['beta'] == 0, torch.tensor(-1e3).to(marginal_device), result_target['beta'])

        
        # greedy_rewards, ent= self.pcfg.sample_crf(grad=result_target['span_marginals'].sum(-1), reward=input_pack["reward"], argmax=True, num_samples=1)
        if sample_mode == "crf" or sample_mode == "a2c":
            # print()
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
        rv_greedy = torch.tensor([[s[1] for s in b] for b in greedy_rewards])
        

        c_nll = mode == "nll"
        c_rlonly = mode == "rlonly"
        c_rl = mode == "rl"
        c_a2c = mode == "a2c"
        c_ta2c = mode == "ta2c"
        c_ta2c_devnorm = mode == "ta2c_devnorm"
        c_ta2c_rules = mode == "ta2c_rules"
        c_ta2c_inside = mode == "ta2c_inside"
        c_a2c_v0 = mode == "a2c_v0"
        c_tavg = mode == "tavg"
        # print("nll, rlonly, rl, a2c", c_nll, c_rlonly, c_rl, c_a2c)
        assert sum([c_nll, c_rlonly, c_rl, c_a2c, c_ta2c, c_ta2c_devnorm, c_ta2c_rules, c_ta2c_inside, c_a2c_v0, c_tavg]) == 1, "only one mode can be selected"
        # set default value for statistics
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
                # adv_ss = torch.clamp(adv_ss, min=-5, max=5)
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
            rv = torch.tensor([[s[1] for s in b] for b in rewards])
            logp = torch.stack([torch.stack([s[0]  for s in b], dim=0) for bid, b in enumerate(rewards)], dim=0).cpu()
            v0 = result_target["v0"]
            assert torch.all(v0<150), f"v0 should be less than 150, {v0}"
            rw = (rv.cpu() - v0.unsqueeze(-1).cpu())
            rw_normalized = rw.detach()
            loss_a2c = (-logp.cpu() * rw_normalized).mean(-1)
            loss = rl_coeff * loss_a2c.cpu() + -result_target["partition"].cpu()
            stat_a2c_loss = loss_a2c.mean(-1)
        elif c_ta2c_rules:
            logp_ss = torch.stack([torch.stack([s[4] for s in b], dim=0) for b in rewards], dim=0).cpu()
            adv_ss = torch.stack([torch.stack([s[5] for s in b], dim=0) for b in rewards], dim=0).cpu().detach()
            ent_ss = torch.stack([torch.stack([s[6] for s in b], dim=0) for b in rewards], dim=0).cpu()
            # print(logp_ss, adv_ss)
            # input()
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
        stat_best_rv = rv_greedy.squeeze(-1).mean(-1)
        stat_nll = -result_target["partition"].mean()
        stat_logppl = stat_nll/maxlen
        stat_entropy = ent.mean(-1)
        stat_rw = rw.std(-1).mean(-1)
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
