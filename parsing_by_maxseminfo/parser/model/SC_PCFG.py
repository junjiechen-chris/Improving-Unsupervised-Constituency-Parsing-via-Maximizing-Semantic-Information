from ast import Not
import torch
import torch.nn as nn
from parsing_by_maxseminfo.parser.modules.res import ResLayer
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.utils.checkpoint import checkpoint as ckp

from parsing_by_maxseminfo.parser.pcfgs.simple_pcfg import MySPCFGFaster
# from parser.pcfgs.simple_pcfg import SimplePCFG_Triton_Batch


def checkpoint(func):
    def wrapper(*args, **kwargs):
        return ckp(func, *args, **kwargs)

    return wrapper

class Simple_C_PCFG(nn.Module):
    def __init__(self, args, dataset):
        super(Simple_C_PCFG, self).__init__()
        # self.pcfg = SimplePCFG_Triton_Batch()

        self.device = dataset.device
        self.args = args
        self.NT = args.NT
        self.T = args.T
        self.V = len(dataset.word_vocab)
        self.s_dim = args.s_dim
        self.z_dim = args.z_dim
        self.enc_dim = args.h_dim

        # self.r = args.r_dim
        self.word_emb_size = args.w_dim
        # rule_dim = self.s_dim
        
        self.entropy  = args.entropy if hasattr(args, 'entropy') else False
        self._para_init()
        self._initialize()  

        ## root

    def _para_init(self):
        rule_dim = self.s_dim
        self.root_emb = nn.Parameter(torch.randn(1, self.s_dim))
        input_dim = self.s_dim + self.z_dim

        #terms
        self.term_mlp = nn.Sequential(nn.Linear(input_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      nn.Linear(self.s_dim, self.V)
        )

        self.rule_state_emb = nn.Parameter(torch.randn(self.NT+self.T, self.s_dim))

        
        self.left_mlp = nn.Sequential(nn.Linear(rule_dim,rule_dim),nn.ReLU()) 
        self.right_mlp = nn.Sequential(nn.Linear(rule_dim,rule_dim),nn.ReLU())
        self.parent_mlp1 =  nn.Sequential(nn.Linear(input_dim, self.s_dim),
                                          nn.ReLU(),
                                      )
    
        self.enc_emb = nn.Embedding(self.V, 512)
        self.enc_rnn = nn.LSTM(512, 512, bidirectional=True, num_layers=1, batch_first=True)
        self.enc_out = nn.Linear(512 * 2, self.z_dim * 2)


    

    def _initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, input, evaluating=False, **kwargs):
        x = input['word']
        b, n = x.shape[:2]
        seq_len = input['seq_len']

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
            z = mean.new(b, mean.size(1)).normal_(0,1)
            z = (0.5 * lvar).exp() * z + mean

        def roots():
            roots = (self.root_emb @ self.rule_state_emb[:self.NT].t())
            roots = roots.log_softmax(-1)
            return roots.expand(b, roots.shape[-1])

        def terms():
            term_emb = self.rule_state_emb[self.NT:].unsqueeze(0).expand(
                b, self.T, self.s_dim
            )            


            z_expand = z.unsqueeze(1).expand(b, self.T, self.z_dim)
            term_emb = torch.cat([term_emb, z_expand], -1)
            term_prob = self.term_mlp(term_emb).log_softmax(-1)
            return term_prob.gather(-1, x.unsqueeze(1).expand(b, self.T, x.shape[-1])).transpose(-1, -2)


        def rules():
            nonterm_emb = self.rule_state_emb[:self.NT].unsqueeze(0).expand(
                b, self.NT, self.s_dim
            )
            z_expand = z.unsqueeze(1).expand(
                b, self.NT, self.z_dim
            )
            nonterm_emb2 = torch.cat([nonterm_emb, z_expand], -1)

            parent1 = self.parent_mlp1(nonterm_emb2) + nonterm_emb

            left = torch.einsum('bnr, mr -> bmn', parent1, (self.left_mlp(self.rule_state_emb) + self.rule_state_emb))
            right =  torch.einsum('bnr, mr -> bmn', parent1, (self.right_mlp(self.rule_state_emb) + self.rule_state_emb))


            left = left.softmax(-2)
            right = right.softmax(-2)

            left_m =  left[:, :self.NT, :].contiguous()
            left_p =  left[:, self.NT:, :].contiguous()
            
            right_m = right[:, :self.NT, :].contiguous()
            right_p = right[:, self.NT:, :].contiguous()
            
            return (left_m, left_p, right_m, right_p)

        root, unary, (left_m, left_p, right_m, right_p) = roots(), terms(), rules()

        return {'unary': unary,
                'root': root,
                'left_m': left_m,
                'right_m': right_m,
                'left_p': left_p,
                'right_p' : right_p,
                'kl': kl(mean, lvar).sum(1)}

    def loss(self, input):
        rules = self.forward(input)
        result =  self.pcfg._inside(rules=rules, lens=input['seq_len'])
        loss =  (-result['partition'] + rules['kl']).mean()
        return loss



    def evaluate(self, input, decode_type, **kwargs):
        rules = self.forward(input)
        if decode_type == 'viterbi':
            assert NotImplementedError

        elif decode_type == 'mbr':
            return self.pcfg.decode(rules=rules, lens=input['seq_len'], viterbi=False, mbr=True)
        else:
            raise NotImplementedError
        


class SCPCFGPairwise(Simple_C_PCFG):
    def __init__(self, args, vocab_size):
        super(Simple_C_PCFG, self).__init__()
        self.pcfg = MySPCFGFaster()
        # self.device = device
        self.args = args

        self.NT = args.NT
        self.T = args.T
        self.V = vocab_size

        self.s_dim = args.s_dim
        self.rule_dim = self.s_dim
        self.z_dim = args.z_dim

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
        assert decode_type == "mbr", "currently only the mbr decoding mode is supported"
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


class SCPCFGOT(SCPCFGPairwise):

    def __init__(self, args, vocab_size, span_repr_mode, langstr):
        super().__init__(args, vocab_size)
        self.wvmodel = None



class SCPCFGFidxedCostReward(SCPCFGOT):
    @torch.enable_grad()
    def loss(
        self,
        input_pack,
        run_wd=True,
        dropout=0.0,
        mode=None,
        rl_coeff=1. ,
        rl_len_norm = False,
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
        result_target = self.pcfg._inside(
            rules=rules_target,
            lens=input_pack["seq_len"],
            span_dist=True,
            allow_grad=run_wd,
            span_mask=None,
            dropout=dropout,
            include_unary=include_unary
        )
        if supervised_mode:
            raise NotImplementedError("Supervised experiment not supported")
        
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
            # loss = -result_target["partition"]
            loss = -result_target["partition"] + rules_target['kl']
        elif c_rlonly:
            loss = rl_coeff * loss_pg
        elif c_rl:
            # loss_pg/=(maxlen) # logp scaling () normalize by sequence length
            # loss = rl_coeff * loss_pg + -result_target["partition"].cpu()
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
            "flag_rl_len_norm": rl_len_norm,
        }

    def train(self, mode=True):
        super().train(mode)
        self.pcfg.train(mode)


