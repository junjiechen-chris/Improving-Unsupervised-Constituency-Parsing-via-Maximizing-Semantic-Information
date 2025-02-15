# from turtle import pd
from calendar import c
import multiprocessing
from re import split
import select
from typing import final

# from numpy import repeat
from itertools import repeat

from sympy import beta

from parsing_by_maxseminfo.parser.pcfgs.pcfgs import PCFG_base
from parsing_by_maxseminfo.parser.pcfgs.fn import (
    checkpoint_nonreentrant,
    stripe,
    diagonal_copy_,
    checkpoint,
    diagonal,
    stripe_add_,
)
import torch
from parsing_by_maxseminfo.parser.triton.fn import _merge, _log_then_diagonal_copy_


import pdb


class SimplePCFG_Triton(PCFG_base):
    def __init__(self):
        super(SimplePCFG_Triton, self).__init__()

    def loss(self, rules, lens):
        return self._inside(rules, lens)

    @torch.enable_grad()
    def _inside(
        self,
        rules,
        lens,
        mbr=False,
        viterbi=False,
        marginal=False,
        s_span=None,
        entropy=False,
    ):
        assert viterbi is not True
        # B, L, r_p
        unary = rules["unary"].clone()
        # B, L, r_m
        root = rules["root"].exp()

        # r_m, r_m
        L = rules["left_m"]
        R = rules["right_m"]
        # r_p, r_p
        L_p = rules["left_p"]
        R_p = rules["right_p"]
        LR = torch.cat([L, R], dim=-1)
        # breakpoint()
        r_p = unary.shape[-1]
        r_m = L.shape[1]

        batch, N, *_ = unary.shape
        N += 1
        # for estimating marginals.
        if s_span is None:
            span_indicator = unary.new_zeros(batch, N, N).requires_grad_(mbr)
        else:
            span_indicator = s_span
            if mbr or viterbi:
                span_indicator = span_indicator.detach().clone().requires_grad_(True)
            unary += diagonal(span_indicator, w=1).unsqueeze(-1)

        # normalizer = unary.new_zeros(batch, N, N).fill_(-1e9)

        with torch.no_grad():
            unary_max = unary.max(-1)[0]

        unary = (unary - unary_max.unsqueeze(-1)).exp()
        unary = torch.einsum("bnp, pq -> bnq", unary, torch.cat([L_p, R_p], dim=-1))

        alpha_c = unary.new_zeros(batch, N, N, 2, r_m)
        alpha_c = _log_then_diagonal_copy_(unary, unary_max, alpha_c)

        # w: span width
        for w in range(2, N):
            n = N - w
            normalizer = alpha_c.new_zeros(batch, n)
            out, normalizer = _merge(normalizer, diagonal(span_indicator, w), alpha_c)
            if w < N - 1:
                out = torch.einsum("blr, rq -> blq", out, LR)
                alpha_c = _log_then_diagonal_copy_(out, normalizer, alpha_c)

        logZ = (
            torch.einsum("bnr, br -> b", out, root) + 1e-9
        ).log() + normalizer.squeeze(1)

        if not mbr and not viterbi:
            return {"partition": logZ}

        elif marginal:
            logZ.sum().backward()
            return {"marginal": span_indicator.grad}

        else:
            return {
                "prediction": self._get_prediction(
                    logZ, span_indicator, lens, mbr=True
                ),
                "partition": logZ,
            }


class SimplePCFG_Triton_Batch(PCFG_base):
    def __init__(self):
        super(SimplePCFG_Triton_Batch, self).__init__()

    def loss(self, rules, lens):
        return self._inside(rules, lens)

    @torch.enable_grad()
    def _inside(
        self,
        rules,
        lens,
        mbr=False,
        viterbi=False,
        marginal=False,
        s_span=None,
        entropy=False,
    ):
        assert viterbi is not True
        # B, L, r_p
        unary = rules["unary"].clone()
        # B, L, r_m
        root = rules["root"].exp()

        # r_m, r_m
        L = rules["left_m"]
        R = rules["right_m"]
        # r_p, r_p
        L_p = rules["left_p"]
        R_p = rules["right_p"]
        LR = torch.cat([L, R], dim=-1)
        r_p = unary.shape[-1]
        r_m = L.shape[-2]
        # breakpoint()
        batch, N, *_ = unary.shape
        N += 1
        # for estimating marginals.
        if s_span is None:
            span_indicator = unary.new_zeros(batch, N, N).requires_grad_(mbr)
        else:
            span_indicator = s_span
            if mbr or viterbi:
                span_indicator = span_indicator.detach().clone().requires_grad_(True)
            unary += diagonal(span_indicator, w=1).unsqueeze(-1)

        # normalizer = unary.new_zeros(batch, N, N).fill_(-1e9)

        with torch.no_grad():
            unary_max = unary.max(-1)[0]

        unary = (unary - unary_max.unsqueeze(-1)).exp()

        unary = torch.einsum("bnp, bpq -> bnq", unary, torch.cat([L_p, R_p], dim=-1))

        alpha_c = unary.new_zeros(batch, N, N, 2, r_m)

        alpha_c = _log_then_diagonal_copy_(unary, unary_max, alpha_c)

        # w: span width
        for w in range(2, N):
            n = N - w
            normalizer = alpha_c.new_zeros(batch, n)

            out, normalizer = _merge(normalizer, diagonal(span_indicator, w), alpha_c)

            if w < N - 1:
                out = torch.einsum("blr, brq -> blq", out, LR)
                alpha_c = _log_then_diagonal_copy_(out, normalizer, alpha_c)

        logZ = (
            torch.einsum("bnr, br -> b", out, root) + 1e-9
        ).log() + normalizer.squeeze(1)

        if not mbr and not viterbi:
            return {"partition": logZ}

        elif marginal:
            logZ.sum().backward()

            return {"marginal": span_indicator.grad}

        else:
            return {
                "prediction": self._get_prediction(
                    logZ, span_indicator, lens, mbr=True
                ),
                "partition": logZ,
            }


class MyPCFG_Triton(PCFG_base):
    def __init__(self):
        super(MyPCFG_Triton, self).__init__()


        
        pass


    def loss(self, rules, lens):
        return self._inside(rules, lens)

    def _get_span_distribution(
        self, logZ, tocopy_array, span_indicator, lens, allow_grad=True
    ):
        NT = tocopy_array[0].shape[-1]
        B, N = span_indicator.shape[:2]
        assert logZ.requires_grad
        assert not span_indicator.requires_grad
        # logZ.sum().backward(
        #     retain_graph=True, create_graph=allow_grad, inputs=tocopy_array
        # )

        grad = torch.autograd.grad(
            logZ.sum(), tocopy_array, create_graph=False, retain_graph=True
        )

        # grad = [t.grad for t in tocopy_array]
        def grad_correction(g, t):
            assert torch.all(t >= 0)
            # print((t>1e-3).sum())
            assert torch.all(torch.logical_and(t >= 0, t <= 1))
            return (
                ((g * (t > 0).float()).log() - (t + 1e-9).log())
                .logsumexp(-1)
                .exp()
                .unsqueeze(-1)
            )

        grad_corrected = [grad_correction(g, t) for g, t in zip(grad, tocopy_array)]

        marginals = span_indicator.new_zeros(B, N, N, NT)  # .requires_grad_()
        for w in range(2, marginals.shape[1]):
            # print(w, tocopy_array[w - 2].grad)
            diagonal_copy_(
                marginals,
                grad_corrected[w - 2],
                w,
            )

        assert not torch.isnan(marginals).any(), "gradient contains nan"
        # for t in tocopy_array:
        # del t.grad
        tocopy_array = marginals
        # marginals = marginals.sum(-1)
        # marginals.sum().backward()
        # print([t.grad for t in tocopy_array[:2]])
        # assert torch.all(torch.isclose(marginals, span_indicator.grad))
        return marginals, tocopy_array

    @torch.enable_grad()
    def _inside(
        self,
        rules,
        lens,
        mbr=False,
        viterbi=False,
        marginal=False,
        s_span=None,
        entropy=False,
        span_dist=False,
        allow_grad=False,
        span_mask=None,
    ):
        assert viterbi is not True
        # B, L, r_p
        unary = rules["unary"].clone()
        # B, L, r_m
        root = rules["root"].exp()

        # r_m, r_m
        L = rules["left_m"]
        R = rules["right_m"]
        # r_p, r_p
        L_p = rules["left_p"]
        R_p = rules["right_p"]
        LR = torch.cat([L, R], dim=-1)
        # breakpoint()
        r_p = unary.shape[-1]
        r_m = L.shape[1]

        batch, N, *_ = unary.shape
        N += 1
        # for estimating marginals.
        if s_span is None:
            span_indicator = unary.new_zeros(batch, N, N).requires_grad_(mbr)
        else:
            span_indicator = s_span
            if mbr or viterbi:
                span_indicator = span_indicator.detach().clone().requires_grad_(True)
            unary += diagonal(span_indicator, w=1).unsqueeze(-1)

        # normalizer = unary.new_zeros(batch, N, N).fill_(-1e9)

        with torch.no_grad():
            unary_max = unary.max(-1)[0]

        unary = (unary - unary_max.unsqueeze(-1)).exp()
        unary = torch.einsum("bnp, pq -> bnq", unary, torch.cat([L_p, R_p], dim=-1))

        alpha_c = unary.new_zeros(batch, N, N, 2, r_m)
        alpha_c = _log_then_diagonal_copy_(unary, unary_max, alpha_c)

        tocopy_array = []
        # w: span width
        for w in range(2, N):
            n = N - w
            normalizer = alpha_c.new_zeros(batch, n)
            out, normalizer = _merge(normalizer, diagonal(span_indicator, w), alpha_c)
            tocopy_array.append(out)
            if w < N - 1:
                out = torch.einsum("blr, rq -> blq", out, LR)
                alpha_c = _log_then_diagonal_copy_(out, normalizer, alpha_c)
        logZ = (
            torch.einsum("bnr, br -> b", out, root) + 1e-9
        ).log() + normalizer.squeeze(1)

        if span_dist:
            marginals, _ = self._get_span_distribution(
                logZ, tocopy_array, span_indicator, lens, allow_grad
            )
            # alpha_c.copy_(alpha_c_clone)
            return {
                # "partition": logZ,
                "span_marginals": marginals,
                "span_indicator": span_indicator,
                "rule_logp": rules,
                "beta": alpha_c,
                "tocopy_array": tocopy_array,
                "alpha_c": alpha_c,
            }

        if not mbr and not viterbi:
            return {"partition": logZ}

        elif marginal:
            logZ.sum().backward()
            return {"marginal": span_indicator.grad}

        else:
            return {
                "prediction": self._get_prediction(
                    logZ, span_indicator, lens, mbr=True
                ),
                "partition": logZ,
            }


class MySPCFGFaster:
    def __init__(self):
        self.training = True

    def train(self, mode):
        self.training = mode

    @staticmethod
    def sample_recursive(l, r, grad, reward, argmax=False, epsilon=0., span_score = None, v_est = None, gae_gamma = 0.95):
        # assert span_score is None ^ v_est is None,  f"span_score and v_est must be provided together, but got {span_score, v_est}"

        # print(span_score[0], v_est[0])
        assert span_score is None or torch.all(span_score == v_est), "span score and vest is typed"

        import numpy as np
        # print(l, r)
        assert len(grad.shape) == 2, "grad should be 2d"
        w = r - l
        if w==1: 
            # v_est[l, r] = 0
            return 0, 0, [(l, r)], 0, [], 0 # logprob, reward
        if w == 2:
            # v_est[l, r] = reward[l, r]
            return 0, reward[l, r], [(l, r)], 0, [(torch.tensor(0.), torch.tensor(0., device=span_score.device), torch.tensor(0.))] if span_score is not None else [], 0 # logprob, reward
        p = []
        for u in range(l + 1, r):
            p.append(grad[l, u] + grad[u, r])

        unnorm_p = torch.torch.stack(p, dim=0)
        # selection_p = torch.nn.functional.softmax(unnorm_p, 0 )
        log_selection_p = torch.nn.functional.log_softmax(unnorm_p, 0)        
        selection_p = log_selection_p.exp()
        # if epsilon > 0:
        #     selection_p = selection_p * (1-epsilon) + epsilon/w
        #     log_selection_p = log_selection_p + np.log(1-epsilon) + np.log(1/len(p))

        if not argmax:
            split = torch.multinomial(selection_p , 1).item()
        else:
            split = selection_p.max(0)[1].item()
        split_logp = log_selection_p[split]
        split = l + split + 1

        lp, lr, ls, le, lrl, lgae= MySPCFGFaster.sample_recursive(l, split, grad, reward, argmax, epsilon=epsilon, span_score=span_score, v_est=v_est)
        rp, rr, rs, re, rrl, rgae = MySPCFGFaster.sample_recursive(split, r, grad, reward, argmax, epsilon=epsilon, span_score=span_score, v_est=v_est)
        assert split_logp <= 0, f"split logp should be negative, but got {split_logp, p}"
        logprob = split_logp + lp + rp
        rwad = reward[l, r] + lr + rr
        entropy = - (selection_p * log_selection_p).sum() 
        
        if span_score is not None:
            adv = reward[l, r] + span_score[l, split] + span_score[split, r] - span_score[l, r]
            gae = adv + gae_gamma * (lgae+rgae)
            

        # print(f"({l, r}): {lrl, rrl}")
        # print("logp, reward", logprob, rwad)
        return logprob, rwad, ls+rs+[(l, r)], entropy, [(split_logp, gae, entropy)] + lrl + rrl if span_score is not None else [], gae if span_score is not None else 0 #lr + rr + reward[l, r]
    
    def sample_recursive_pcfg(self, prod_tuple, l, r, nt, precomputed, reward, argmax=False, epsilon=0., LR=False, tau = 1., span_score = None, v_est = None, gae_gamma=0.95):
        # LR = False -> left LR = True -> right
        import numpy as np
        # print(l, r)
        # print(beta.shape, prod.shape)
        # print(l, r, nt)
        # print("recursive len precomputed", len(precomputed))

        # c_a2c = span_score is not None
        # if c_a2c: assert v_est is not None, "v_est must be provided to save the updated value"

        beta, unary = precomputed
        _, _, _, rules = prod_tuple

        L_m = rules["left_m"]#+ 1e-12
        R_m = rules["right_m"]#+ 1e-12
        L_p = rules["left_p"]#+ 1e-12
        R_p = rules["right_p"]#+ 1e-12


        NT, PT = L_m.shape[0], L_p.shape[0]

        def matmul_kernel(x, y):
            # print("/y.shape)
            assert len(y.shape) == 1, f"y must be 1d {y.shape}"
            # assert torch.all
            y_max = y.max(0)[0]
            # print("matmul shape", x.shape, y.shape) 
            return ((((y - y_max).exp() @ x) + 1e-9) ).log() + y_max
        


        # def flood(x):
        #     assert len(x.shape) == 1, "x must be 1d"
        #     return torch.stack([, x], dim=-1).logsumexp(-1)


        w = r - l
        # if w==1: return (L_p.log() + unary[l].unsqueeze(1)).logsumexp(0) if not LR else (R_p.log() + unary[l].unsqueeze(1)).logsumexp(0), 0, [(l, r)], 0
        if w==1: 
            # print(unary[l].max(0))
            # v_est[l, r] = 0
            if not LR:
                converted_logp = ((L_p+1e-9).log() + unary[l].unsqueeze(1)).logsumexp(0)
            else:
                converted_logp = ((R_p+1e-9).log() + unary[l].unsqueeze(1)).logsumexp(0)
            assert len(converted_logp.shape) == 1, f"converted shape must have (NT), {converted_logp.shape}"
            return torch.tensor(0. , device=beta.device), 0, [(l, r)], converted_logp, [], 0 
        if w == 2:
            if w == len(beta)-1:
                raise NotImplementedError("this line should not be touched")
                return beta[l, r], reward[l, r], [(l, r)], 0
            # v_est[l, r] = reward[l, r]
            # return (L_m.log() + beta[l, r].unsqueeze(1)).logsumexp(0) if not LR else (R_m.log() + beta[l, r].unsqueeze(1)).logsumexp(0), reward[l, r], [(l, r)], 0 # logprob, reward
            if not LR:
                converted_logp = ((L_m+1e-9).log() + beta[l, r].unsqueeze(1)).logsumexp(0)
            else:
                converted_logp = ((R_m+1e-9).log() + beta[l, r].unsqueeze(1)).logsumexp(0)
            assert len(converted_logp.shape) == 1, f"converted shape must have (NT), {converted_logp.shape}"
            return torch.tensor(0., device=beta.device), reward[l, r], [(l, r)], converted_logp, [(torch.tensor(0., device=span_score.device), torch.tensor(0., device=span_score.device), torch.tensor(0., device=span_score.device))] if span_score is not None else [], 0  # logprob, reward
        p = []
        # v_est_candidate = []

        # working_beta = beta[:, :, nt]
        # working_prod = prod[nt, :, :]

        assert len(beta.shape) == 3, "grad should be 3d"

        for u in range(l + 2, r-1):
            p.append(matmul_kernel(L_m[:, nt], beta[l, u]) + matmul_kernel(R_m[:, nt], beta[u, r]))
        p = [matmul_kernel(L_p[:, nt], unary[l]) + matmul_kernel(R_m[:, nt], beta[l+1, r])] + p
        p.append(matmul_kernel(R_p[:, nt], unary[r-1]) + matmul_kernel(L_m[:, nt], beta[l, r-1]))
                
        unnorm_p_span = [it.flatten().squeeze(0) for it in p]
        assert len(unnorm_p_span)>0, f"the span_selection must have more than one choices"
        # print(unnorm_p_span)
        unnorm_p_span = torch.stack(unnorm_p_span, dim=0)
        # unnorm_p_span = torch.clamp(unnorm_p_span, min=-1e9)
        # print(unnorm_p_span)
        assert torch.all(~torch.isnan(unnorm_p_span)), "unnorm_p_span must be nonnan"
        assert torch.all(~torch.isinf(unnorm_p_span)), "unnorm_p_span must be noninf"
        # assert torch.isclose(unnorm_p_span.logsumexp(0), beta[l, r, nt]), f"unnorm_p_span must be equal to beta {unnorm_p_span.logsumexp(0), beta[l, r, nt], beta[l, r]}" 
        selection_logp = torch.log_softmax(unnorm_p_span/tau, 0)
        # print(selection_p)

        if not argmax:
            # print(selection_logp.exp())
            span_split = torch.multinomial(selection_logp.exp() , 1).item()
            action_logp = selection_logp[span_split]
            # v_est[l, r] = v_est_candidate[span_split]
            span_split += l +1
            if span_split == l+1:
                nt_logits = (L_p[:, nt]+1e-9).unsqueeze(-1).log() + (R_m[:, nt]+1e-9).unsqueeze(-2).log() + unary[l].unsqueeze(-1) + beta[l+1, r].unsqueeze(-2) 
            elif span_split == r-1:
                nt_logits = (L_m[:, nt]+1e-9).unsqueeze(-1).log() + (R_p[:, nt]+1e-9).unsqueeze(-2).log() + beta[l, r-1].unsqueeze(-1) + unary[r-1].unsqueeze(-2) 
            else:
                nt_logits = (L_m[:, nt]+1e-9).unsqueeze(-1).log() + (R_m[:, nt]+1e-9).unsqueeze(-2).log() + beta[l, span_split].unsqueeze(-1) + beta[span_split, r].unsqueeze(-2)
            nt_selection_logp = torch.log_softmax(nt_logits.flatten()/tau, 0)
            nt_choice = torch.multinomial(nt_selection_logp.exp(), 1).item()
            # action_logp = action_logp #+ nt_selection_logp[nt_choice]
        else:
            span_split = selection_logp.max(0)[1].item()
            action_logp = selection_logp[span_split]

            span_split += l +1
            
            if span_split == l+1:
                nt_logits = L_p[:, nt].unsqueeze(-1).log() + R_m[:, nt].unsqueeze(-2).log() + unary[l].unsqueeze(-1) + beta[l+1, r].unsqueeze(-2) 
            elif span_split == r-1:
                nt_logits = L_m[:, nt].unsqueeze(-1).log() + R_p[:, nt].unsqueeze(-2).log() + beta[l, r-1].unsqueeze(-1) + unary[r-1].unsqueeze(-2) 
            else:
                nt_logits = L_m[:, nt].unsqueeze(-1).log() + R_m[:, nt].unsqueeze(-2).log() + beta[l, span_split].unsqueeze(-1) + beta[span_split, r].unsqueeze(-2)
            nt_selection_logp = torch.log_softmax(nt_logits.flatten()/tau, 0)
            nt_choice = nt_selection_logp.max(0)[1].item()
            # action_logp = action_logp #+ nt_selection_logp[nt_choice]



        if span_split == l+1:
            nt_1, nt_2 = divmod(nt_choice, NT)
            assert nt_1<PT and nt_2<NT, f"nt1 - PT, nt2-NT {nt_1, nt_2}"
        elif span_split == r-1:
            nt_1, nt_2 = divmod(nt_choice, PT)
            assert nt_1<NT and nt_2<PT, f"nt1 - NT, nt2-PT {nt_1, nt_2}"
        else:
            nt_1, nt_2 = divmod(nt_choice, NT)
            assert nt_1<NT and nt_2<NT, f"nt1, 2 must both be NTs {nt_1, nt_2}"
        

        lp, lr, ls, le, lrl, lgae = self.sample_recursive_pcfg(prod_tuple, l, span_split, nt_1, precomputed, reward, argmax, epsilon=epsilon, LR=False, tau=tau, span_score=span_score)
        rp, rr, rs, re, rrl, rgae = self.sample_recursive_pcfg(prod_tuple, span_split, r, nt_2, precomputed, reward, argmax, epsilon=epsilon, LR=True, tau=tau, span_score=span_score)
        assert torch.all(lp<=0.1) and torch.all(rp<=0.1), f"lp, rp should be in the log scale {l, span_split, r} lp: {torch.max(lp)} rp: {torch.max(rp)}"
        # assert split_logp <= 0, f"split logp should be negative, but got {split_logp, p}"

        # print(beta[l, span_split].max(), beta[span_split, r].max(), working_prod.max())
        assert w<len(beta), "span width must be smaller than len(beta)"
        # logprob = action_logp
        # if w == len(beta)-1:
        #     logprob = lp + rp
        #     assert not torch.any(torch.isnan(logprob))
        # elif not LR:
        #     logprob = matmul_kernel(L_m, lp+rp)
        #     # logprob = (L_m.log() + (lp + rp).unsqueeze(-1)).logsumexp(0)
        #     assert not torch.any(torch.isnan(logprob))
        # else: 
        #     # logprob = (R_m.log() + (lp+rp).unsqueeze(-1)).logsumexp(0) 
        #     logprob = matmul_kernel(R_m, lp+rp)
        #     assert not torch.any(torch.isnan(logprob))

        logprob = action_logp + lp + rp
        # print("logprob:", logprob)
        rwad = reward[l, r] + lr + rr
        # print("reward", l, r, rwad)
        # entropy = torch.tensor(0., device=beta.device) #- (selection_p * log_selection_p).sum() + le + re
        # print(f"lp, rp", lp.shape, rp.shape)
        if w==len(beta)-1:
            converted_logp = le+re
        elif not LR:
            # converted_logp = ((L_m+1e-9).log() + (le + re).unsqueeze(-1)).logsumexp(0)
            converted_logp = matmul_kernel(L_m, le+re)
            # logprob = (L_m.log() + (lp + rp).unsqueeze(-1)).logsumexp(0)
            assert not torch.any(torch.isnan(converted_logp))
        elif LR:
            converted_logp = matmul_kernel(R_m, le+re)
            # converted_logp = ((R_m+1e-9).log() + (le + re).unsqueeze(-1)).logsumexp(0)
            assert not torch.any(torch.isnan(converted_logp))

        # print("logp, reward", logprob, rwad)
        # return logprob, rwad, ls+rs+[(l, r)], None #lr + rr + reward[l, r]
        
        if span_score is not None:
            # print("shapes", reward.shape, span_score.shape, l, r, span_split)
            l_ss = span_score[l, span_split,nt_1] if span_split-l>1 else 0
            r_ss = span_score[span_split, r, nt_2] if r-span_split>1 else 0

            adv = reward[l, r] + l_ss + r_ss - span_score[l, r, nt]#.mean(-1)
            gae = adv + gae_gamma * (lgae+rgae)
            
        return logprob, rwad, ls+rs+[(l, r)], converted_logp, [(action_logp + nt_selection_logp[nt_choice], gae, torch.tensor(0., device=beta.device))] + lrl + rrl if span_score is not None else [], gae if span_score is not None else 0 #lr + rr + reward[l, r]
    
    def sample_pcfg(self, precomputed, reward, num_samples=1, argmax=False, is_grad_in_log=False, log_normalization = None, epsilon=0., prod =None, root_rule = None, tau=1., span_score=None):
        # grad = grad.to("cpu")
        # print("sampling from pcfg", beta.shape, prod.shape)
        # print("precomputed", len(precomputed))
        assert prod is not None, "production rules must be applied"
        assert root_rule is not None, "root rule must be applied"

        # if not is_grad_in_log:
            # grad = torch.clamp(grad.log(), -30)
        
        # assert log_normalization is None
        assert epsilon == 0, "epsilon is causing worse performance"
        # if log_normalization is None:
        #     log_norm = torch.zeros(grad.shape[0])
        # else:
        #     log_norm = log_normalization

        assert len(root_rule.shape) == 2, "root rule should be 2d, one prescribing the batch number, the other the nt number"



        root_rule_exp = root_rule.exp()
        # print("root_rule inspection", root_rule)
        # input()


        reward = reward.to("cpu")
        batch, N, _ = reward.shape
        N-=1
        samples = []
        starting_nt = torch.multinomial(root_rule_exp.repeat(1, num_samples).reshape(batch*num_samples, -1), 1).flatten()
        # print("beta, unary shape", precomputed[0].shape, precomputed[1].shape)
        # input()
        nt_idx = 0
        for b in range(batch):
            sample = []
            for _ in range(num_samples):
                # print(f"sampling for {b} batch {_} sample")
                out = self.sample_recursive_pcfg(prod, 0, N, starting_nt[nt_idx], [a[b] for a in precomputed], reward[b], argmax, epsilon=epsilon, tau=tau, span_score=span_score[b] if span_score is not None else None)
                out = list(out)[:-1]
                # out[3]
                nt_idx += 1
                # out[3] = out[3] - log_norm[b]
                if span_score is not None:
                    # print(out[-1])
                    single_step_actions, single_step_advs, single_step_ent = zip(*out[-1])
                    single_step_actions = torch.stack(single_step_actions, dim=0)
                    single_step_advs = torch.stack(single_step_advs, dim=0)
                    single_step_ent = torch.stack(single_step_ent, dim=0)
                    # print(single_step_actions, single_step_advs)
                    out = out[:-1]
                    out += [single_step_actions, single_step_advs, single_step_ent]
                out[3] = (out[3]+root_rule[b]).logsumexp(0) #- log_norm[b]
                sample.append(out)
            samples.append(sample)
        return samples




    def sample(self, grad, reward, num_samples=1, argmax=False, is_grad_in_log=False, log_normalization = None, epsilon=0., span_score = None, v_est = None):
        grad = grad.to("cpu")

        if not is_grad_in_log:
            grad = torch.clamp(grad.log(), -30)
        
        assert log_normalization is None
        assert epsilon == 0, "epsilon is causing worse performance"
        # if log_normalization is None:
        #     log_norm = torch.zeros(grad.shape[0])
        # else:
        #     log_norm = log_normalization
        reward = reward.to("cpu")
        batch, N, _ = reward.shape
        N-=1
        samples = []
        for b in range(batch):
            sample = []
            for _ in range(num_samples):
                # print(f"sampling for {b} batch {_} sample")
                out = self.sample_recursive(0, N, grad[b], reward[b], argmax, epsilon=epsilon, span_score = span_score[b] if span_score is not None else None, v_est=v_est[b] if v_est is not None else None)
                out = list(out)
                out = out[:-1] # remove the gae return
                if span_score is not None:
                    # print(out[-2])
                    single_step_actions, single_step_advs, single_step_ent = zip(*out[-1])
                    single_step_actions = torch.stack(single_step_actions, dim=0)
                    single_step_advs = torch.stack(single_step_advs, dim=0)
                    single_step_ent = torch.stack(single_step_ent, dim=0)
                    # print(single_step_actions, single_step_advs)
                    out = out[:-1]
                    out += [single_step_actions, single_step_advs, single_step_ent]
                # out[3] = out[3] - log_norm[b]
                sample.append(out)
            samples.append(sample)
        return samples

    @staticmethod
    def _sample_worker(N, grad, reward):
        samples = []
        for g, r in zip(grad, reward):
            samples.append(MySPCFGFaster.sample_recursive(0, N, g, r))
        return samples

    def sample_batch(self, grad, reward, num_samples=1, argmax=False, is_grad_in_log=False, log_normalization = None, epsilon=0., num_worker = 6):
        grad = grad.to("cpu")
        assert argmax == False, "argmax is not supported for batch sampling"
        reward = reward.to("cpu")
        batch, N, _ = reward.shape
        N-=1
        chunk_size = ((batch*num_samples)//(num_worker))+1
        # samples_input = [ for b in range(batch) for _ in range(num_samples)]
        grad_input = grad.unsqueeze(1).expand(-1, num_samples, -1, -1).reshape(-1, N+1, N+1)
        reward_input = reward.unsqueeze(1).expand(-1, num_samples, -1, -1).reshape(-1, N+1, N+1)
        grad_chunk = grad_input.split(chunk_size, dim=0)
        reward_chunk = reward_input.chunk(chunk_size, dim=0)
        assert len(grad_chunk) == num_worker
        with multiprocessing.Pool(num_worker) as p:
            samples = p.starmap(MySPCFGFaster._sample_worker, 
                                zip(repeat(N), grad_chunk, reward_chunk))
        samples = [s for sample in samples for s in sample]
        return samples
            
    def sample_LR(self, L, R, reward, num_samples=1, argmax=False, log_normalization = None):
        grad = grad.to("cpu")

        grad = L + R

        log_norm = log_normalization
        reward = reward.to("cpu")
        batch, N, _ = reward.shape
        N-=1
        samples = []
        for b in range(batch):
            sample = []
            for _ in range(num_samples):
                # print(f"sampling for {b} batch {_} sample")
                out = self.sample_recursive(0, N, grad[b], reward[b], argmax)
                out = list(out)
                out[3] = out[3] - log_norm[b]
                sample.append(out)
            samples.append(sample)
        return samples

        


    @torch.no_grad()
    def _cky_zero_order(self, marginals, lens):
        N = marginals.shape[-1]
        s = marginals.new_zeros(*marginals.shape).fill_(-1e9)
        p = marginals.new_zeros(*marginals.shape).long()
        diagonal_copy_(s, diagonal(marginals, 1), 1)
        for w in range(2, N):
            n = N - w
            starts = p.new_tensor(range(n))
            if w != 2:
                Y = stripe(s, n, w - 1, (0, 1))
                Z = stripe(s, n, w - 1, (1, w), 0)
            else:
                Y = stripe(s, n, w - 1, (0, 1))
                Z = stripe(s, n, w - 1, (1, w), 0)
            X, split = (Y + Z).max(2)
            x = X + diagonal(marginals, w)
            diagonal_copy_(s, x, w)
            diagonal_copy_(p, split + starts.unsqueeze(0) + 1, w)

        def backtrack(p, i, j):
            if j == i + 1:
                return [(i, j)]
            split = p[i][j]
            ltree = backtrack(p, i, split)
            rtree = backtrack(p, split, j)
            return [(i, j)] + ltree + rtree

        p = p.tolist()
        lens = lens.tolist()
        spans = [backtrack(p[i], 0, length) for i, length in enumerate(lens)]
        return spans

    def build_alpha_chart(self, grad):
        
        batch, N, _ = grad.shape
        grad = grad.unsqueeze(-1)
        alpha = grad.new_zeros(batch, N, N, 1).fill_(0)
        # print(alpha.shape, grad.shape, stripe(grad, N-2 , 1, (0, 1)).shape)
        unary = stripe(grad, N-1, 1, (0, 1)).clone()
        if any(unary.flatten() > 0):
            diagonal_copy_(alpha, unary.squeeze(2), 1,)
        diagonal_copy_(alpha, stripe(grad, N-2 , 1, (0, 2)).squeeze(2), 2,)
        for w in range(3, N):

            n = N - w
            # x = terms.new_zeros(3, batch, n, NT).fill_(-1e9)

            Y = stripe(alpha, n, w - 1, (0, 1)).clone()
            Z = stripe(alpha, n, w - 1, (1, w), 0).clone()
            # X = stripe(grad, n, 1, (0, w)).clone()
            X = diagonal(grad, w).clone().unsqueeze(-2)

            # print(X.shape, Y.shape, Z.shape)

            x = (X + Y + Z).logsumexp(-2)

            diagonal_copy_(
                alpha,
                x,
                w,
            )
        return alpha.squeeze(-1)
    
    def build_alpha_chart_a2c(self, grad, reward, span_score):
        # print(grad.shape, reward.shape)

        batch, N, _ = grad.shape
        grad = grad.unsqueeze(-1)
        reward = reward.unsqueeze(-1)
        alpha = grad.new_zeros(batch, N, N, 1).fill_(0)
        v_est = grad.new_zeros(batch, N, N, 1).fill_(0)
        # span_score = span_score.unsqueeze(-1)
        entropy = grad.new_zeros(batch, N, N, 1).fill_(0)
        # print(alpha.shape, grad.shape, stripe(grad, N-2 , 1, (0, 1)).shape)
        unary = stripe(grad, N-1, 1, (0, 1)).clone()
        if any(unary.flatten() > 0):
            diagonal_copy_(alpha, unary.squeeze(2), 1,)
        diagonal_copy_(alpha, stripe(grad, N-2 , 1, (0, 2)).squeeze(2), 2,)
        diagonal_copy_(v_est, stripe(reward, N-2, 1, (0, 2)).clone().squeeze(2), 2,)

        for w in range(3, N):

            n = N - w
            # x = terms.new_zeros(3, batch, n, NT).fill_(-1e9)

            Y = stripe(alpha, n, w - 1, (0, 1)).clone()
            Z = stripe(alpha, n, w - 1, (1, w), 0).clone()
            X = diagonal(grad, w)#stripe(grad, n, 1, (0, w)).clone()

            RW_Y = stripe(v_est, n, w - 1, (0, 1)).clone()
            RW_Z = stripe(v_est, n, w - 1, (1, w), 0).clone()
            RW_X = diagonal(reward, w).clone()

            # print(X.shape, Y.shape, Z.shape)

            x = (X.unsqueeze(-2) + Y + Z).logsumexp(-2)



            log_conditional_split = (Y+Z).log_softmax(-2)

            conditional_split = log_conditional_split.exp()
            # print(Y.shape, conditional_split.shape, RW_Y.shape, RW_X.shape)
            assert torch.isclose(conditional_split.sum(-2), torch.ones_like(conditional_split)[:, :, 0, :]).all()

            ent = - (conditional_split * log_conditional_split).sum(-2)

            v_est_split = (RW_Y + RW_Z)
            # print(conditional_split[0].squeeze(-1), v_est_split[0].squeeze(-1))
            # print("v_est:", v_est_split[0].squeeze(-1), RW_X[0].squeeze(-1), (v_est_split * conditional_split)[0].squeeze(-1))
            # input()
            v_est_current = ((v_est_split * conditional_split)).sum(-2) + RW_X


            diagonal_copy_(
                alpha,
                x,
                w,
            )

            diagonal_copy_(
                v_est,
                v_est_current,
                w,
            )

            diagonal_copy_(
                entropy,
                ent,
                w,
            )

        return alpha.squeeze(-1), v_est.squeeze(-1), entropy.squeeze(-1)
    
    def sample_crf(self, grad, reward, num_samples=1, argmax=False, epsilon=0., span_score = None):
        grad = torch.clamp(grad.log(), -30)
        if span_score is None:
            alpha = self.build_alpha_chart(grad)
            return self.sample(alpha, reward, num_samples=num_samples, is_grad_in_log=True, epsilon=epsilon, span_score=span_score), self.sample(alpha, reward, num_samples=1, is_grad_in_log=True, argmax=True, span_score=None), self.entropy(alpha), v_est if span_score is not None else None
        else:
            alpha, v_est, ent = self.build_alpha_chart_a2c(grad, reward, span_score)
            return self.sample(alpha, reward, num_samples=num_samples, is_grad_in_log=True, epsilon=epsilon, span_score=v_est, v_est=v_est), self.sample(alpha, reward, num_samples=1, is_grad_in_log=True, argmax=True, span_score=None, v_est=None), self.entropy(alpha), v_est if span_score is not None else None

    def sample_pcfg_entrance(self, beta, reward, num_samples=1, argmax=False, is_grad_in_log=False, log_normalization = None, epsilon=0., prod =None, root_rule = None, tau=1., span_score = None):
        # grad = torch.clamp(grad.log(), -30)
        # alpha = self.build_alpha_chart(grad\)
        # beta, reward, num_samples=1, argmax=False, is_grad_in_log=False, log_normalization = None, epsilon=0., prod =None, root_rule = None
        return self.sample_pcfg(beta, reward, num_samples=num_samples, argmax=False, epsilon=0., prod =prod, root_rule = root_rule, tau=tau, span_score=span_score), self.sample_pcfg(beta, reward, num_samples=1, argmax=True, prod =prod, root_rule = root_rule, tau=tau), torch.tensor([0], dtype=torch.float)#, self.entropy(alpha)
    
    def entropy(self, alpha):
        batch, N, _ = alpha.shape
        alpha = alpha.unsqueeze(-1)
        entropy = alpha.new_zeros(batch, N, N, 1).fill_(0)
        # print(alpha.shape, grad.shape, stripe(grad, N-2 , 1, (0, 1)).shape)
        # diagonal_copy_(alpha, stripe(grad, N-2 , 1, (0, 2)).squeeze(2), 2,)
        for w in range(3, N):
            # alpha is well-defined for

            n = N - w
            # x = terms.new_zeros(3, batch, n, NT).fill_(-1e9)

            Y = stripe(alpha, n, w - 1, (0, 1)).clone()
            Z = stripe(alpha, n, w - 1, (1, w), 0).clone()
            logprob = (Y+Z).log_softmax(-2)
            prob = logprob.exp()

            EY = stripe(entropy, n, w - 1, (0, 1)).clone()
            EZ = stripe(entropy, n, w - 1, (1, w), 0).clone()
            prev_entropy = (EY + EZ)


            ent = ((prev_entropy - logprob) * prob).sum(-2)
            # print(ent)

            diagonal_copy_(
                entropy,
                ent,
                w,
            )
        ent = entropy.squeeze(-1)[:, 0, N-1]
        assert (ent>=0).all(), "entropy should be non-negative"
        return ent

        return entropy.squeeze(-1)[:, 0, N-1]



    @torch.enable_grad()
    def decode(
        self, rules, lens, viterbi=False, mbr=False, allow_grad=False, span_mask=None
    ):
        return self._inside(
            rules=rules,
            lens=lens,
            viterbi=viterbi,
            mbr=mbr,
            allow_grad=allow_grad,
            span_mask=span_mask,
        )

    def _get_span_distribution(
        self, logZ, tocopy_array, span_indicator, lens, include_unary, allow_grad=True
    ):
        # batch, seq_len = s.shape[:2]
        # ! skip the seq_len>=3 requirement
        # if seq_len >= 3:

        NT = tocopy_array[1].shape[-1]
        B, N = span_indicator.shape[:2]
        assert logZ.requires_grad
        assert not span_indicator.requires_grad
        if not include_unary:
            tocopy_array = tocopy_array[1:]
        grads = torch.autograd.grad(logZ.sum(), tocopy_array, create_graph=allow_grad)
        corrected_grads = [torch.maximum(g.new_zeros(*g.shape), g) for g in grads]
        marginals = span_indicator.new_zeros(B, N, N, NT)  # .requires_grad_()
        if include_unary:
            diagonal_copy_(marginals, (corrected_grads[0].sum(-1, keepdim=True).expand(-1, -1, NT)/NT), 1)
            corrected_grads = corrected_grads[1:]

        w_start = 2 #if not include_unary else 1

        for w in range(w_start, marginals.shape[1]):
            # print(w, tocopy_array[w - 2].grad)
            diagonal_copy_(
                marginals,
                corrected_grads[w - w_start],
                w,
            )
        assert torch.all(marginals >= 0), "marginals contains negative values"
        assert not torch.isnan(marginals).any(), "gradient contains nan"
        # for t in tocopy_array:
        # del t.grad
        tocopy_array = marginals
        marginals = marginals
        # marginals.sum().backward()
        # print([t.grad for t in tocopy_array[:2]])
        # assert torch.all(torch.isclose(marginals, span_indicator.grad))
        return marginals, tocopy_array

    def _get_prediction(self, logZ, span_indicator, lens, mbr=False, allow_grad=False):
        batch, seq_len = span_indicator.shape[:2]
        prediction = [[] for _ in range(batch)]
        # to avoid some trivial corner cases.
        if seq_len >= 3:
            assert logZ.requires_grad
            logZ.sum().backward(retain_graph=allow_grad, create_graph=allow_grad)
            marginals = span_indicator.grad
            # print(marginals.sum([1, 2]))
            if mbr:
                return self._cky_zero_order(marginals.detach(), lens), marginals
            else:
                viterbi_spans = marginals.nonzero().tolist()
                for span in viterbi_spans:
                    prediction[span[0]].append((span[1], span[2]))
        return prediction, marginals


    # def build_beta_from_tocopy(self, tocopy):
    #     slen = max([t.shape[1] for t in tocopy])+1
    #     batch,_,  NT = tocopy[0].shape
    #     tocopy_sorted_asending = sorted(tocopy, key=lambda x: x.shape[1])  
    #     padded_tocopy = [t.pad((0, 0, 0, slen-t.shape[1], 0, 0)) for t in tocopy_sorted_asending]

        # beta = tocopy[0].new_zeros(batch, slen, slen, NT)





    @torch.enable_grad()
    def _inside(
        self,
        rules,
        lens,
        mbr=False,
        viterbi=False,
        span_dist=False,
        allow_grad=False,
        span_mask=None,
        dropout=0.0,
        dropout_pt=0.0,
        s_span=None,
        include_unary=False,
        reward = None, 
    ):
        assert dropout == 0.0, "dropout must be set to 0."
        assert viterbi is not True
        unary = rules["unary"].clone()
        root = rules["root"].exp()
        # print("root.shape", root.shape)
        assert torch.isclose(root.sum(-1) , torch.tensor(1., device = root.device)).all(), "root must be a valid distribution"

        # r_m, r_m
        L = rules["left_m"]
        R = rules["right_m"]
        # r_p, r_p
        L_p = rules["left_p"]
        R_p = rules["right_p"]

        assert torch.all(~torch.isnan(L))
        assert torch.all(~torch.isnan(R))
        assert torch.all(~torch.isnan(R_p))
        assert torch.all(~torch.isnan(L_p))
        

        if len(rules["left_m"].shape) == 2:
            # r_m, r_m
            L = L.unsqueeze(0)
            R = R.unsqueeze(0)
            # r_p, r_p
            L_p = L_p.unsqueeze(0)
            R_p = R_p.unsqueeze(0)


        r_p = unary.shape[-1]
        r_m = L.shape[1]
        # print(unary.shape, L.shape)
        

        # # 3d binary rule probabilities tensor decomposes to three 2d matrices after CP decomposition.
        # H = rules['head'].clone()  # (batch, NT, r) r:=rank
        # L = rules['left'].clone()  # (batch, NT+T, r)
        # R = rules['right'].clone()  # (batch, NT+T, r)

        # T = unary.shape[-1]
        # S = L.shape[-2]
        # NT = S - T
        # r = L.shape[-1]

        # L_term = L[:, NT:, ...].contiguous()
        # L_nonterm = L[:, :NT, ...].contiguous()
        # R_term = R[:, NT:, ...].contiguous()
        # R_nonterm = R[:, :NT, ...].contiguous()

        # H = H.transpose(-1, -2)
        # (m_A, r_A) + (m_B, r_B) -> (r_A, r_B)
        # H_L = torch.matmul(H, L_nonterm)
        # H_R = torch.matmul(H, R_nonterm)

        c_vest = reward is not None

        def transform(x, y):
            """
            :param x: shape (batch, n, T)
            :return: shape (batch, n, r)
            """
            return torch.matmul(x, y)

        # @checkpoint
        # @checkpoint_nonreentrant
        def merge(Y, Z, y, z, indicator, additive_mask = None):
            """
            :param Y: shape (batch, n, w, N)
            :param Z: shape (batch, n, w, N)
            :return: shape (batch, n, x)
            """
            # contract dimension w.
            Y = (Y + 1e-9).log() + y.unsqueeze(-1)
            Z = (Z + 1e-9).log() + z.unsqueeze(-1)
            assert torch.all(~torch.isnan(Y + Z))
            b_n_r = (Y + Z).logsumexp(-2) + indicator
            if additive_mask is not None:
                b_n_r = (b_n_r + additive_mask).clamp(min=-1e9)
            normalizer = b_n_r.max(-1)[0]
            logbnr = b_n_r
            b_n_r = (b_n_r - normalizer.unsqueeze(-1)).exp()
            # print(b_n_r)
            assert torch.all(~torch.isnan(b_n_r))
            assert torch.all(b_n_r >= 0) and torch.all(b_n_r <= 1)
            return b_n_r, normalizer, logbnr
        
        def transform_kernel(x, y, v):
            # print(y.shape)
            # assert len(y.shape) == 1, "y must be 1d"
            assert y.shape == v.shape, f"beta must have the same shape as v, {y.shape, v.shape}"
            assert ((x<=1) & (x>=0)).all(), f"rules must be normalized"
            # assert torch.all
            y_max = y.max(-1)[0].unsqueeze(-1)
            # print('transform_kernel x.shape', x.shape)
            # print((y - y_max).exp()[0, :, 0, 0], v[0, :, 0, 0])
            # input()
            # print("matmul shape", x.shape, y.shape, v.shape) 
            assert torch.all(~torch.isnan((y - y_max).exp()))
            assert torch.all(~torch.isnan(v))
            assert torch.all(~torch.isnan(x))
            return (((((y - y_max).exp() * v) @ x[None, :, :, :]) + 1e-9).log() + y_max)

        @torch.no_grad()
        def compute_vest(RW_Y, RW_Z, RW_X, BETA_Y, BETA_Z, BETA_X, LS, RS ,L, R):
            #deal with nt enumerations
            assert torch.all(~torch.isnan(L)) and torch.all(~torch.isnan(R))
            assert torch.all(~torch.isnan(BETA_Y)) and torch.all(~torch.isnan(BETA_Z)) and torch.all(~torch.isnan(BETA_X))
            assert torch.all(~torch.isnan(RW_Y)) and torch.all(~torch.isnan(RW_Z)) and torch.all(~torch.isnan(RW_X))
            l = transform_kernel(L, BETA_Y, RW_Y)
            l_1 = LS
            # l_1 = transform_kernel(L, BETA_Y, BETA_Y.new_ones(*BETA_Y.shape))
            r = transform_kernel(R, BETA_Z, RW_Z)
            # r_1 = transform_kernel(R, BETA_Z, BETA_Z.new_ones(*BETA_Z.shape))
            r_1 = RS

            # print("rw_y", RW_Y[0, :, :, 0])
            # # print("rw_z", RW_Z[0, :, 0])
            # print("beta y", BETA_Y[0, :, :, 0] )
            # print("beta z", BETA_Z[0, :, :, 0] )
            # # print("rw", RW_X[0])
            # print("l_primitive, vet accumulate", (l)[0, :, :, 0])
            # print("l, vet accumulate", (l+r_1)[0, :, :, 0])
            # print("r, vet accumulate", (r+l_1)[0, :, :, 0])

            l = (l+r_1).logsumexp(-2)# - BETA_X
            r = (r+l_1).logsumexp(-2)# - BETA_X
            v_est = (torch.stack([l, r], dim=0).logsumexp(0) - BETA_X).exp() + RW_X.unsqueeze(-1) 
            # print("vest", v_est[0, :, 0])
            # v_est = l.exp() + r.exp() + RW_X.unsqueeze(-1) 




            assert torch.all(~torch.isnan(l)) 
            assert torch.all(~torch.isnan(r))
            assert torch.all(v_est>=0)
            return v_est



        batch, N, *_ = unary.shape
        N += 1
        tocopy_array = []
        tocopy_array.append(unary) # first term is unary potentials.
        beta = unary.new_zeros(batch, N, N, L.shape[2]).fill_(-1e9)
        # diagonal_copy_(unary, beta, w=1)

        if span_mask is not None:
            assert span_mask.shape == (batch, N, N) 
            span_mask = span_mask.unsqueeze(-1)
            # multiplicative_mask = span_mask
            # print(span_mask)
            additive_span_mask = (1-span_mask) * -1e9


        # for estimating marginals.
        if s_span is None:
            span_indicator = unary.new_zeros(batch, N, N).requires_grad_(mbr)
        else:
            span_indicator = s_span
            if mbr or viterbi:
                span_indicator = span_indicator.detach().clone().requires_grad_(True)
            unary += diagonal(span_indicator, w=1).unsqueeze(-1)

        normalizer = unary.new_zeros(batch, N, N).fill_(-1e9)
        norm = unary.max(-1)[0]
        diagonal_copy_(normalizer, norm, w=1)
        unary = (unary - norm.unsqueeze(-1)).exp()
        # print("unary", unary.shape, L_p.shape, R_p.shape)
        left_term = transform(unary, L_p)
        right_term = transform(unary, R_p)

        # for caching V^{T}s_{i,k} and W^{T}s_{k+1,j} as described in paper to decrease complexities.
        left_s = unary.new_zeros(batch, N, N, L.shape[2]).fill_(-1e9)
        right_s = unary.new_zeros(batch, N, N, L.shape[2]).fill_(-1e9)
        if c_vest:
            v_est = unary.new_zeros(batch, N, N, L.shape[2]).fill_(0)
        else: 
            v_est = None


        # print("pre diag copy", left_s.shape, left_term.shape)
        diagonal_copy_(left_s, left_term, w=1)
        diagonal_copy_(right_s, right_term, w=1)

        # w: span width
        for w in range(2, N):
            # n: the number of spans of width w.
            n = N - w
            Y = stripe(left_s, n, w - 1, (0, 1))
            Z = stripe(right_s, n, w - 1, (1, w), 0)
            Y_normalizer = stripe(normalizer, n, w - 1, (0, 1))
            Z_normalizer = stripe(normalizer, n, w - 1, (1, w), 0)
            x, x_normalizer, logx = merge(
                Y.clone(),
                Z.clone(),
                Y_normalizer.clone(),
                Z_normalizer.clone(),
                span_indicator[:, torch.arange(n), w + torch.arange(n)].unsqueeze(-1),
                additive_mask=additive_span_mask[:, torch.arange(n), w + torch.arange(n)] if span_mask is not None else None
            )
            tocopy_array.append(logx)
            keep_mask = torch.nn.functional.dropout(
                x.new_ones(x.shape), p=dropout, training=self.training
            ).bool()
            x = (x * keep_mask)/(1-dropout)  # $x.masked_fill(dropout_mask, 0)

            if c_vest:
                vest_y = stripe(v_est, n, w - 1, (0, 1))
                vest_z = stripe(v_est, n, w - 1, (1, w), 0)
                reward_x = diagonal(reward, w)
                beta_y = stripe(beta, n, w - 1, (0, 1))
                beta_z = stripe(beta, n, w - 1, (1, w), 0) 
                v = compute_vest(vest_y, vest_z, reward_x, beta_y, beta_z, logx, (Y+1e-9).log()+Y_normalizer.unsqueeze(-1), (Z+1e-9).log()+Z_normalizer.unsqueeze(-1), L, R)
                # input()
                diagonal_copy_(v_est, v, w)

            if w + 1 < N:
                left_x = transform(x, L)
                right_x = transform(x, R)
                diagonal_copy_(left_s, left_x, w)
                diagonal_copy_(right_s, right_x, w)
                # if span_mask is not None:
                    # left_s = left_s * multiplicative_mask
                    # right_s = right_s * multiplicative_mask
                diagonal_copy_(normalizer, x_normalizer, w)
                diagonal_copy_(beta, logx, w)
            else:
                final_m = x
                diagonal_copy_(beta, logx, w)

        final = (final_m + 1e-9).squeeze(1).log() + root.log()
        logZ = final.logsumexp(-1) + x_normalizer.squeeze(-1)
        if c_vest:
            assert v.shape[1] == 1, f"final v must be form (b, 1, NT), {v.shape}"
            # print(v.shape, root.shape)
            v0 = (v.squeeze(1) * root).sum(-1)

            # v0 = (v_est[torch.arange(batch), 0, lens] * root).sum(-1)

        # def get_beta(tocopy_array, span_indicator):
        #     NT = tocopy_array[0].shape[-1]
        #     B, N = span_indicator.shape[:2]
        #     assert logZ.requires_grad
        #     assert not span_indicator.requires_grad
        #     beta = span_indicator.new_zeros(B, N, N, NT)  # .requires_grad_()
        #     for w in range(2, beta.shape[1]):
        #         # print(w, tocopy_array[w - 2].grad)
        #         diagonal_copy_(
        #             beta,
        #             tocopy_array[w - 2],
        #             w,
        #     )
        #     return beta
        
        # print(v_est[0, :, :, 0])
        # print("v0", v0)
        # input()

        if mbr or viterbi:
            prediction, marginals = self._get_prediction(
                logZ, span_indicator, lens, mbr=mbr, allow_grad=False
            )
            return {"prediction": prediction, "partition": logZ}
        elif span_dist:
            marginals, _ = self._get_span_distribution(
                logZ, tocopy_array, span_indicator, lens, allow_grad=allow_grad, include_unary=include_unary
            )
            return {
                "partition": logZ,
                "span_marginals": marginals,
                "tocopy_array": tocopy_array,
                # "beta": get_beta(tocopy_array, span_indicator),
                "left": left_s,
                "right": right_s,
            }
        else:
            return {"partition": logZ, "beta": beta, "unary": rules["unary"], "v_est": v_est, "v0": v0}
