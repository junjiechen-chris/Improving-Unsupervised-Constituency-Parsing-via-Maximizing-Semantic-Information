from ast import Not
from uu import Error
import torch
import torch.nn as nn
from parsing_by_maxseminfo.utils.utils import compute_oneside_matching_with_spanoverlap_score_competing,  compute_oneside_matching_with_spanoverlap_score_merged
from parsing_by_maxseminfo.parser.modules.res import ResLayer
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from ..pcfgs.pcfg import PCFG, FasterMyPCFG, MyPCFG


class CompoundPCFG(nn.Module):
    def __init__(self, args, dataset):
        super(CompoundPCFG, self).__init__()

        self.pcfg = PCFG()
        self.device = dataset.device
        self.args = args
        self.NT = args.NT
        self.T = args.T
        self.V = len(dataset.word_vocab)

        self.s_dim = args.s_dim
        self.z_dim = args.z_dim
        self.w_dim = args.w_dim
        self.h_dim = args.h_dim

        self._init_params()
        self._initialize()

    def _init_params(self):
        self.term_emb = nn.Parameter(torch.randn(self.T, self.s_dim))
        self.nonterm_emb = nn.Parameter(torch.randn(self.NT, self.s_dim))
        self.root_emb = nn.Parameter(torch.randn(1, self.s_dim))

        input_dim = self.s_dim + self.z_dim

        self.term_mlp = nn.Sequential(
            nn.Linear(input_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
            nn.Linear(self.s_dim, self.V),
        )

        self.root_mlp = nn.Sequential(
            nn.Linear(input_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
            nn.Linear(self.s_dim, self.NT),
        )

        self.enc_emb = nn.Embedding(self.V, self.w_dim)

        self.enc_rnn = nn.LSTM(
            self.w_dim, self.h_dim, bidirectional=True, num_layers=1, batch_first=True
        )
    
        self.enc_out = nn.Linear(self.h_dim * 2, self.z_dim * 2)
        # self.enc_out_bert = nn.Linear(768, self.z_dim * 2)

        self.NT_T = self.NT + self.T
        self.rule_mlp = nn.Linear(input_dim, (self.NT_T) ** 2)

    def _initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                # torch.nn.init.xavier_uniform_(p)
                torch.nn.init.xavier_uniform_(p)

    def forward(self, input, evaluating=False):
        x = input["word"]
        b, n = x.shape[:2]
        seq_len = input["seq_len"]

        def enc(x):
            x_embbed = self.enc_emb(x)
            x_packed = pack_padded_sequence(
                x_embbed, seq_len.cpu(), batch_first=True, enforce_sorted=False
            )
            h_packed, _ = self.enc_rnn(x_packed)
            padding_value = float("-inf")
            output, lengths = pad_packed_sequence(
                h_packed, batch_first=True, padding_value=padding_value
            )
            h = output.max(1)[0]
            out = self.enc_out(h)
            mean = out[:, : self.z_dim]
            lvar = out[:, self.z_dim :]
            return mean, lvar

        def kl(mean, logvar):
            result = -0.5 * (logvar - torch.pow(mean, 2) - torch.exp(logvar) + 1)
            return result

        mean, lvar = enc(x)
        z = mean

        if not evaluating:
            z = mean.new(b, mean.size(1)).normal_(0, 1)
            z = (0.5 * lvar).exp() * z + mean

        def roots():
            root_emb = self.root_emb.expand(b, self.s_dim)
            root_emb = torch.cat([root_emb, z], -1)
            roots = self.root_mlp(root_emb).log_softmax(-1)
            return roots

        def terms():
            term_emb = self.term_emb.unsqueeze(0).expand(b, self.T, self.s_dim)
            z_expand = z.unsqueeze(1).expand(b, self.T, self.z_dim)
            term_emb = torch.cat([term_emb, z_expand], -1)
            term_prob = self.term_mlp(term_emb).log_softmax(-1)
            return term_prob.gather(
                -1, x.unsqueeze(1).expand(b, self.T, x.shape[-1])
            ).transpose(-1, -2)

        def rules():
            nonterm_emb = self.nonterm_emb.unsqueeze(0).expand(b, self.NT, self.s_dim)
            z_expand = z.unsqueeze(1).expand(b, self.NT, self.z_dim)
            nonterm_emb = torch.cat([nonterm_emb, z_expand], -1)
            rule_prob = self.rule_mlp(nonterm_emb).log_softmax(-1)
            rule_prob = rule_prob.reshape(b, self.NT, self.NT_T, self.NT_T)
            return rule_prob

        root, unary, rule = roots(), terms(), rules()

        return {"unary": unary, "root": root, "rule": rule, "kl": kl(mean, lvar).sum(1)}

    def loss(self, input):
        rules = self.forward(input)
        result = self.pcfg._inside(rules=rules, lens=input["seq_len"])
        loss = (-result["partition"] + rules["kl"]).mean()
        return loss

    def evaluate(self, input, decode_type, **kwargs):
        rules = self.forward(input, evaluating=True)
        if decode_type == "viterbi":
            result = self.pcfg.decode(
                rules=rules, lens=input["seq_len"], viterbi=True, mbr=False
            )
        elif decode_type == "mbr":
            result = self.pcfg.decode(
                rules=rules, lens=input["seq_len"], viterbi=False, mbr=True
            )
        else:
            raise NotImplementedError

        result["partition"] -= rules["kl"]
        return result


class CompoundPCFGPairwise(CompoundPCFG):
    def __init__(self, args, vocab_size):
        super(CompoundPCFG, self).__init__()
        self.pcfg = MyPCFG()
        # self.device = device
        self.args = args
        self.NT = args.NT
        self.T = args.T
        self.V = vocab_size

        self.s_dim = args.s_dim
        self.z_dim = args.z_dim
        self.w_dim = args.w_dim
        self.h_dim = args.h_dim

        self._init_params()
        self._initialize()

    def loss(self, input_target, input_pas):
        raise NotImplementedError("This class serves only as a base class")

class CompoundPCFGOT(CompoundPCFGPairwise):
    def __init__(self, args, vocab_size, span_repr_mode, langstr):
        super().__init__(args, vocab_size)
        self.wvmodel = None        
        if hasattr(self.args, "use_fast_pcfg" ) and  self.args.use_fast_pcfg:
            self.pcfg = FasterMyPCFG()

class CompoundPCFGFixedCostReward(CompoundPCFGOT):
    def __init__(self, args, vocab_size, span_repr_mode, langstr):
        super().__init__(args, vocab_size, span_repr_mode, langstr)
    

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
        rules_target = self.forward(input_pack, bert_mode=False) #No pretrained embedding in release. 
        result_target = self.pcfg._inside(
            rules=rules_target,
            lens=input_pack["seq_len"],
            span_dist=run_wd,
            allow_grad=run_wd,
            span_mask=None,
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
            loss = -result_target["partition"] + rules_target['kl']
        elif c_rlonly:
            loss = rl_coeff * loss_pg
        elif c_rl:
            loss = rl_coeff * loss_pg + (-result_target["partition"].cpu()+ rules_target['kl'].cpu())#-result_target["partition"].cpu()
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
            # "span_marginals": target_span_marginals,
            # "tocopy_array": result_target["tocopy_array"],
        }

    
    def forward(self, input, evaluating=False, bert_mode = "disabled"):
        x = input["word"]
        b, n = x.shape[:2]
        seq_len = input["seq_len"]
        assert bert_mode in ["disabled", "trainable", "frozen"], f"bert_mode {bert_mode} not supported"

        def enc(x):
            x_embbed = self.enc_emb(x)
            x_packed = pack_padded_sequence(
                x_embbed, seq_len.cpu(), batch_first=True, enforce_sorted=False
            )
            h_packed, _ = self.enc_rnn(x_packed)
            padding_value = float("-inf")
            output, lengths = pad_packed_sequence(
                h_packed, batch_first=True, padding_value=padding_value
            )
            h = output.max(1)[0]
            out = self.enc_out(h)
            mean = out[:, : self.z_dim]
            lvar = out[:, self.z_dim :]
            return mean, lvar
        
        def kl(mean, logvar):
            result = -0.5 * (logvar - torch.pow(mean, 2) - torch.exp(logvar) + 1)
            return result

        if bert_mode == "disabled":
            mean, lvar = enc(x)
            z = mean
        else:
            raise NotImplementedError
            # mean, lvar = enc_bert(input['encoded_input'])
            # z = mean

        if not evaluating:
            z = mean.new(b, mean.size(1)).normal_(0, 1)
            z = (0.5 * lvar).exp() * z + mean

        def roots():
            root_emb = self.root_emb.expand(b, self.s_dim)
            root_emb = torch.cat([root_emb, z], -1)
            roots = self.root_mlp(root_emb).log_softmax(-1)
            return roots

        def terms():
            term_emb = self.term_emb.unsqueeze(0).expand(b, self.T, self.s_dim)
            z_expand = z.unsqueeze(1).expand(b, self.T, self.z_dim)
            term_emb = torch.cat([term_emb, z_expand], -1)
            term_prob = self.term_mlp(term_emb).log_softmax(-1)
            return term_prob.gather(
                -1, x.unsqueeze(1).expand(b, self.T, x.shape[-1])
            ).transpose(-1, -2)

        def rules():
            nonterm_emb = self.nonterm_emb.unsqueeze(0).expand(b, self.NT, self.s_dim)
            z_expand = z.unsqueeze(1).expand(b, self.NT, self.z_dim)
            nonterm_emb = torch.cat([nonterm_emb, z_expand], -1)
            rule_prob = self.rule_mlp(nonterm_emb).log_softmax(-1)
            rule_prob = rule_prob.reshape(b, self.NT, self.NT_T, self.NT_T)
            return rule_prob

        root, unary, rule = roots(), terms(), rules()

        return {"unary": unary, "root": root, "rule": rule, "kl": kl(mean, lvar).sum(1)}

