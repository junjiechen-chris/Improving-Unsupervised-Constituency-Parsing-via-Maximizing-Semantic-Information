import torch
import torch.nn as nn
# from parsing_by_maxseminfo.utils.utils import compute_oneside_matching_with_spanoverlap_score_competing, compute_oneside_matching_with_spanoverlap_score_competing_v2, compute_oneside_matching_with_spanoverlap_score_merged, compute_oneside_matching_with_spanoverlap_score_merged_v2
from parsing_by_maxseminfo.parser.modules.res import ResLayer
from ..pcfgs.tdpcfg import TDPCFG, Fastest_TDPCFG, MyTDPCFGFaster, Triton_TDPCFG, MyTDPCFG


class TNPCFG(nn.Module):
    def __init__(self, args, dataset):
        super(TNPCFG, self).__init__()
        self.pcfg = TDPCFG()
        self.device = dataset.device
        self.args = args
        self.NT = args.NT
        self.T = args.T
        self.V = len(dataset.word_vocab)
        self.s_dim = args.s_dim
        self.r = args.r_dim
        self.word_emb_size = args.word_emb_size

        self._init_params()
        # self._initialize()
        ## root

    def _initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def _init_params(self):
        self.root_emb = nn.Parameter(torch.randn(1, self.s_dim))
        self.root_mlp = nn.Sequential(
            nn.Linear(self.s_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
            nn.Linear(self.s_dim, self.NT),
        )

        # terms
        self.term_emb = nn.Parameter(torch.randn(self.T, self.s_dim))
        self.term_mlp = nn.Sequential(
            nn.Linear(self.s_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
            nn.Linear(self.s_dim, self.V),
        )

        self.rule_state_emb = nn.Parameter(torch.randn(self.NT + self.T, self.s_dim))
        rule_dim = self.s_dim
        self.parent_mlp = nn.Sequential(
            nn.Linear(rule_dim, rule_dim), nn.ReLU(), nn.Linear(rule_dim, self.r)
        )
        self.left_mlp = nn.Sequential(
            nn.Linear(rule_dim, rule_dim), nn.ReLU(), nn.Linear(rule_dim, self.r)
        )
        self.right_mlp = nn.Sequential(
            nn.Linear(rule_dim, rule_dim), nn.ReLU(), nn.Linear(rule_dim, self.r)
        )

    def forward(self, input, **kwargs):
        x = input["word"]
        b, n = x.shape[:2]

        def roots():
            roots = self.root_mlp(self.root_emb).log_softmax(-1)
            return roots.expand(b, roots.shape[-1]).contiguous()

        def terms():
            term_prob = self.term_mlp(self.term_emb).log_softmax(-1)
            return term_prob[torch.arange(self.T)[None, None], x[:, :, None]]

        def rules():
            rule_state_emb = self.rule_state_emb
            nonterm_emb = rule_state_emb[: self.NT]
            head = self.parent_mlp(nonterm_emb).log_softmax(-1)
            left = self.left_mlp(rule_state_emb).log_softmax(-2)
            right = self.right_mlp(rule_state_emb).log_softmax(-2)
            head = head.unsqueeze(0).expand(b, *head.shape)
            left = left.unsqueeze(0).expand(b, *left.shape)
            right = right.unsqueeze(0).expand(b, *right.shape)
            return (head, left, right)

        root, unary, (head, left, right) = roots(), terms(), rules()

        return {
            "unary": unary,
            "root": root,
            "head": head,
            "left": left,
            "right": right,
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


class TNPCFGBaseline(TNPCFG):
    def __init__(self, args, vocab_size, device):
        super(TNPCFG, self).__init__()
        self.pcfg = TDPCFG()
        self.device = device
        self.args = args
        self.NT = args.NT
        self.T = args.T
        self.V = vocab_size
        self.s_dim = args.s_dim
        self.r = args.r_dim
        self.word_emb_size = args.word_emb_size

        self._init_params()


class FastTNPCFG(nn.Module):
    def __init__(self, args, dataset):
        super(FastTNPCFG, self).__init__()
        if args.use_triton:
            self.pcfg = Triton_TDPCFG()
        else:
            self.pcfg = Fastest_TDPCFG()
        self.device = dataset.device
        self.args = args
        self.NT = args.NT
        self.T = args.T
        self.V = len(dataset.word_vocab)
        self.s_dim = args.s_dim
        self.r = args.r_dim
        self.word_emb_size = args.word_emb_size

        ## root
    def _init_params(self):
        self.root_emb = nn.Parameter(torch.randn(1, self.s_dim))
        self.root_mlp = nn.Sequential(
            nn.Linear(self.s_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
            # )
            nn.Linear(self.s_dim, self.NT),
        )

        # terms
        # self.term_emb = nn.Parameter(torch.randn(self.T, self.s_dim))
        self.term_mlp = nn.Sequential(
            nn.Linear(self.s_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
            nn.Linear(self.s_dim, self.V),
        )

        self.rule_state_emb = nn.Parameter(torch.randn(self.NT + self.T, self.s_dim))
        rule_dim = self.s_dim
        self.parent_mlp = nn.Sequential(nn.Linear(rule_dim, rule_dim), nn.ReLU())
        self.left_mlp = nn.Sequential(nn.Linear(rule_dim, rule_dim), nn.ReLU())
        self.right_mlp = nn.Sequential(nn.Linear(rule_dim, rule_dim), nn.ReLU())

        self.rank_proj = nn.Parameter(torch.randn(rule_dim, self.r))

        self._initialize()

    def _initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p)

    def forward(self, input, **kwargs):
        x = input["word"]
        b, n = x.shape[:2]

        def roots():
            roots = self.root_mlp(self.root_emb)
            roots = roots.log_softmax(-1)
            return roots.expand(b, roots.shape[-1])

        def terms():
            term_emb = self.rule_state_emb[self.NT :]
            term_prob = self.term_mlp(term_emb).log_softmax(-1)
            return term_prob[torch.arange(self.T)[None, None], x[:, :, None]]
            # term_prob = term_prob.unsqueeze(0).unsqueeze(1).expand(
            #     b, n, self.T, self.V
            # )
            # indices = x.unsqueeze(2).expand(b, n, self.T).unsqueeze(3)
            # term_prob = torch.gather(term_prob, 3, indices).squeeze(3)
            # return term_prob

        def rules():
            rule_state_emb = self.rule_state_emb
            nonterm_emb = rule_state_emb[: self.NT]
            head = self.parent_mlp(nonterm_emb) @ self.rank_proj
            left = self.left_mlp(rule_state_emb) @ self.rank_proj
            right = self.right_mlp(rule_state_emb) @ self.rank_proj
            head = head.softmax(-1)
            left = left.softmax(-2)
            right = right.softmax(-2)
            head = head.unsqueeze(0).expand(b, *head.shape)
            left = left.unsqueeze(0).expand(b, *left.shape)
            right = right.unsqueeze(0).expand(b, *right.shape)
            return (head, left, right)

        root, unary, (head, left, right) = roots(), terms(), rules()

        return {
            "unary": unary,
            "root": root,
            "head": head,
            "left": left,
            "right": right,
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
class TNPCFGPairwise(FastTNPCFG):
    def __init__(self, args, vocab_size):
        super(FastTNPCFG, self).__init__()
        self.pcfg = MyTDPCFGFaster()
        # self.device = device
        self.args = args
        self.NT = args.NT
        self.T = args.T
        self.V = vocab_size
        self.s_dim = args.s_dim
        self.r = args.r_dim
        self.word_emb_size = args.word_emb_size

        self._init_params()

    def loss_target_only(self, input_target):
        # marginal_device = input_target["word"].device
        rules_target = self.forward(input_target)
        result_target = self.pcfg._inside(
            rules=rules_target, lens=input_target["seq_len"], span_dist=False
        )
        return {
            "nll": -result_target["partition"].mean(),
        }

    def loss(self, input_target, input_pas):
        #! Let's assume here that input_target and input_pas has the same batch-size and one entry in input_target corresponds to one entry at the same row in input_pas
        marginal_device = input_target["word"].device
        rules_target = self.forward(input_target)
        result_target = self.pcfg._inside(
            rules=rules_target, lens=input_target["seq_len"], span_dist=True
        )

        target_span_marginals = result_target["span_marginals"] / (
            input_target["seq_len"] - 1
        )[:, None, None].to(marginal_device)

        rules_pas = self.forward(input_pas)
        result_pas = self.pcfg._inside(
            rules=rules_pas, lens=input_pas["seq_len"], span_dist=True
        )
        pas_span_marginals = result_pas["span_marginals"] / (input_pas["seq_len"] - 1)[
            :, None, None
        ].to(marginal_device)

        return {
            "nll": -result_target["partition"].mean(),
            "span_marginals": target_span_marginals,
            "pas_nll": -result_pas["partition"].mean(),
            "pas_span_marginals": pas_span_marginals,
        }

class TNPCFGOT(TNPCFGPairwise):
    def __init__(self, args, vocab_size, span_repr_mode, langstr):
        super().__init__(args, vocab_size)
        if span_repr_mode == "bge-m3":
            from FlagEmbedding import BGEM3FlagModel
            self.wvmodel = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
        elif span_repr_mode == "wv":
            import gensim
            # langstr = kwargs["langstr"]
            print('loading fasttext model for', langstr)
            if langstr == "english":
                self.wvmodel = gensim.models.fasttext.load_facebook_vectors('fasttext/cc.en.300.bin.gz')
            elif langstr == "german":
                self.wvmodel = gensim.models.fasttext.load_facebook_vectors('fasttext/cc.de.300.bin.gz')
            else:
                raise ValueError("wv model only supports english and german")
        elif span_repr_mode in ['em', 'bow']:
            self.wvmodel = None
            pass
        else:
            raise ValueError(f"span_repr_mode {span_repr_mode} not supported")

        if hasattr(self.args, "use_fast_pcfg" ) and  self.args.use_fast_pcfg:
            self.pcfg = MyTDPCFGFaster()
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
        from ipynb.utils import (
            compute_batch_wasserstein_distance_v2,
            compute_batch_wasserstein_distance_geomloss,
            compute_batch_wasserstein_distance_geomloss_gumbel_softmax
        )
        from torch.profiler import profile, record_function, ProfilerActivity

        # with record_function("model_forward_nll"):
        marginal_device = input_target["word"].device
        rules_target = self.forward(input_target)
        result_target = self.pcfg._inside(
            rules=rules_target,
            lens=input_target["seq_len"],
            span_dist=True,
            allow_grad=run_wd,
            span_mask=span_mask,
        )


        if flag_use_ppl_as_nll_loss:
            target_seqlen = input_target['seq_len'].to(marginal_device)
            pas_seqlen = input_pas['seq_len'].to(marginal_device)
        else:
            target_seqlen = input_target['word'].new_ones(input_target['seq_len'].shape)
            pas_seqlen = input_pas['word'].new_ones(input_pas['seq_len'].shape)

        target_span_marginals = result_target["span_marginals"] / (
            input_target["seq_len"] - 1
        )[:, None, None, None].to(marginal_device)
        # target_span_indicator = result_target["span_indicator"]

        rules_pas = self.forward(input_pas)
        result_pas = self.pcfg._inside(
            rules=rules_pas,
            lens=input_pas["seq_len"],
            span_dist=True,
            allow_grad=run_wd,
            span_mask=span_mask,
        )
        pas_span_marginals = result_pas["span_marginals"] / (input_pas["seq_len"] - 1)[
            :, None, None, :None
        ].to(marginal_device)
        # pas_span_indicator = result_pas["span_indicator"]

    #
    # with record_function("model_forward_ot"):
        if run_wd:
            wd_loss_array, wd_array = compute_batch_wasserstein_distance_geomloss_gumbel_softmax(
                self.wvmodel,
                input_target["form"],
                input_pas["form"],
                # input_target["wv"],
                # input_pas["wv"],
                input_pas["seq_len"],
                target_span_marginals,
                pas_span_marginals,
                pasid2groupid_map,
                training=True,
                max_bandwidth=max_bandwidth,
                gumbel_softmax_tau=gumbel_softmax_tau,
                num_samples=32,
                flag_span_repr_mode=flag_span_repr_mode,
                flag_similarity_mode = flag_similarity_mode,
            )
        else:
            wd_loss_array = torch.tensor([0.0])
            wd_array = torch.tensor([0.0])

        return {
            "nll": -(result_target["partition"]/target_seqlen).mean(),
            "span_marginals": target_span_marginals,
            "pas_nll": -(result_pas["partition"]/pas_seqlen).mean(),
            "pas_span_marginals": pas_span_marginals,
            "wd_loss": wd_loss_array.mean(),
            "wd_reward": wd_array,
            # "span_indicator": target_span_indicator,
            # "pas_span_indicator": pas_span_indicator,
            # "rule_logp": result_target["rule_logp"],
            # "tocopy_array": result_target["tocopy_array"],
            # "unmod_span_marginals": result_target["span_marginals"],
        }
    
    
class TNPCFGFixedCostReward(TNPCFGOT):
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
        supervised_mode = False,
        sample_mode = "crf",
    ):
        assert sample_mode == "crf", "This class support only the CRF mode"
        batch_size, maxlen = input_pack["word"].shape
        # marginal_device = input_pack["word"].device
        rules_target = self.forward(input_pack)
        result_target = self.pcfg._inside(
            rules=rules_target,
            lens=input_pack["seq_len"],
            span_dist=run_wd,
            allow_grad=run_wd,
            span_mask=None,
            dropout=dropout,
            # dropout_pt=dropout_pt,
            include_unary=include_unary
        )

        if supervised_mode:
            raise NotImplementedError("supervised training is not supported")


        rewards, greedy_rewards, ent, _ = self.pcfg.sample_crf(grad=result_target["span_marginals"].sum(-1), reward=input_pack["reward"], num_samples=num_samples, epsilon=sample_epsilon)

        assert torch.all(ent >= 0), "entropy should be non-negative, got {}".format(ent)
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
        }

 


    def train(self, mode=True):
        super().train(mode)
        self.pcfg.train(mode)
