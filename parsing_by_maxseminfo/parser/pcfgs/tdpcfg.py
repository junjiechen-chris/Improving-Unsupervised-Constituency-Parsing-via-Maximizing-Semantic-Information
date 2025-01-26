import pdb
from parser.pcfgs.pcfgs import PCFG_base
from parser.pcfgs.fn import (
    checkpoint_nonreentrant,
    stripe,
    diagonal_copy_,
    checkpoint,
    diagonal,
)
from parser.triton.fn import _log_then_diagonal_copy_, _merge
import torch


class TDPCFG(PCFG_base):
    def __init__(self):
        super(TDPCFG, self).__init__()

    def loss(self, rules, lens):
        return self._inside(rules, lens)

    @torch.enable_grad()
    def _inside(self, rules, lens, mbr=False, viterbi=False):
        assert viterbi is not True
        unary = rules["unary"]
        root = rules["root"]

        # 3d binary rule probabilities tensor decomposes to three 2d matrices after CP decomposition.
        H = rules["head"]  # (batch, NT, r) r:=rank
        L = rules["left"]  # (batch, NT+T, r)
        R = rules["right"]  # (batch, NT+T, r)

        T = unary.shape[-1]
        S = L.shape[-2]
        NT = S - T
        # r = L.shape[-1]

        L_term = L[:, NT:, ...].contiguous()
        L_nonterm = L[:, :NT, ...].contiguous()
        R_term = R[:, NT:, ...].contiguous()
        R_nonterm = R[:, :NT, ...].contiguous()

        @checkpoint
        def transform_left_t(x, left):
            """
            :param x: shape (batch, n, T)
            :return: shape (batch, n, r)
            """
            return (x.unsqueeze(-1) + left.unsqueeze(1)).logsumexp(2)

        @checkpoint
        def transform_left_nt(x, left):
            return (x.unsqueeze(-1) + left.unsqueeze(1)).logsumexp(2)

        @checkpoint
        def transform_right_t(x, right):
            return (x.unsqueeze(-1) + right.unsqueeze(1)).logsumexp(2)

        @checkpoint
        def transform_right_nt(x, right):
            return (x.unsqueeze(-1) + right.unsqueeze(1)).logsumexp(2)

        # @checkpoint
        def merge(Y, Z):
            """
            :param Y: shape (batch, n, w, r)
            :param Z: shape (batch, n, w, r)
            :return: shape (batch, n, x)
            """
            # contract dimension w.
            b_n_r = (Y + Z).logsumexp(-2)
            # contract dimension r.
            b_n_x = (b_n_r.unsqueeze(-2) + H.unsqueeze(1)).logsumexp(-1)
            return b_n_x

        batch, N, *_ = unary.shape
        N += 1

        # for estimating marginals.
        span_indicator = unary.new_zeros(batch, N, N).requires_grad_(mbr)

        left_term = transform_left_t(unary, L_term)
        right_term = transform_right_t(unary, R_term)

        s = unary.new_zeros(batch, N, N, NT).fill_(-1e9)
        # for caching V^{T}s_{i,k} and W^{T}s_{k+1,j} as described in paper to decrease complexities.
        left_s = unary.new_zeros(batch, N, N, L.shape[2]).fill_(-1e9)
        right_s = unary.new_zeros(batch, N, N, L.shape[2]).fill_(-1e9)

        diagonal_copy_(left_s, left_term, w=1)
        diagonal_copy_(right_s, right_term, w=1)

        # w: span width
        for w in range(2, N):
            # n: the number of spans of width w.
            n = N - w
            Y = stripe(left_s, n, w - 1, (0, 1))
            Z = stripe(right_s, n, w - 1, (1, w), 0)
            x = merge(Y.clone(), Z.clone())
            x = x + span_indicator[:, torch.arange(n), w + torch.arange(n)].unsqueeze(
                -1
            )
            if w + 1 < N:
                left_x = transform_left_nt(x, L_nonterm)
                right_x = transform_right_nt(x, R_nonterm)
                diagonal_copy_(left_s, left_x, w)
                diagonal_copy_(right_s, right_x, w)
            diagonal_copy_(s, x, w)

        final = s[torch.arange(batch), 0, lens] + root
        logZ = final.logsumexp(-1)

        if not mbr and not viterbi:
            return {"partition": logZ}

        else:

            return {
                "prediction": self._get_prediction(
                    logZ, span_indicator, lens, mbr=True
                ),
                "partition": logZ,
            }


class Fastest_TDPCFG(PCFG_base):
    def __init__(self):
        super(Fastest_TDPCFG, self).__init__()

    def loss(self, rules, lens):
        return self._inside(rules, lens)

    @torch.enable_grad()
    def _inside(
        self, rules, lens, mbr=False, viterbi=False, marginal=False, s_span=None
    ):
        assert viterbi is not True
        unary = rules["unary"].clone()
        root = rules["root"].clone()

        # 3d binary rule probabilities tensor decomposes to three 2d matrices after CP decomposition.
        H = rules["head"].clone()  # (batch, NT, r) r:=rank
        L = rules["left"].clone()  # (batch, NT+T, r)
        R = rules["right"].clone()  # (batch, NT+T, r)

        T = unary.shape[-1]
        S = L.shape[-2]
        NT = S - T
        # r = L.shape[-1]

        L_term = L[:, NT:, ...].contiguous()
        L_nonterm = L[:, :NT, ...].contiguous()
        R_term = R[:, NT:, ...].contiguous()
        R_nonterm = R[:, :NT, ...].contiguous()

        H = H.transpose(-1, -2)
        # (m_A, r_A) + (m_B, r_B) -> (r_A, r_B)
        H_L = torch.matmul(H, L_nonterm)
        H_R = torch.matmul(H, R_nonterm)

        def transform(x, y):
            """
            :param x: shape (batch, n, T)
            :return: shape (batch, n, r)
            """
            return torch.matmul(x, y)

        @checkpoint
        def merge(Y, Z, y, z, indicator):
            """
            :param Y: shape (batch, n, w, r)
            :param Z: shape (batch, n, w, r)
            :return: shape (batch, n, x)
            """
            # contract dimension w.
            Y = (Y + 1e-9).log() + y.unsqueeze(-1)
            Z = (Z + 1e-9).log() + z.unsqueeze(-1)
            b_n_r = (Y + Z).logsumexp(-2) + indicator
            normalizer = b_n_r.max(-1)[0]
            b_n_r = (b_n_r - normalizer.unsqueeze(-1)).exp()
            return b_n_r, normalizer

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

        normalizer = unary.new_zeros(batch, N, N).fill_(-1e9)
        norm = unary.max(-1)[0]
        diagonal_copy_(normalizer, norm, w=1)
        unary = (unary - norm.unsqueeze(-1)).exp()
        left_term = transform(unary, L_term)
        right_term = transform(unary, R_term)

        # for caching V^{T}s_{i,k} and W^{T}s_{k+1,j} as described in paper to decrease complexities.
        left_s = unary.new_zeros(batch, N, N, L.shape[2]).fill_(-1e9)
        right_s = unary.new_zeros(batch, N, N, L.shape[2]).fill_(-1e9)

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
            x, x_normalizer = merge(
                Y.clone(),
                Z.clone(),
                Y_normalizer.clone(),
                Z_normalizer.clone(),
                span_indicator[:, torch.arange(n), w + torch.arange(n)].unsqueeze(-1),
            )

            if w + 1 < N:
                left_x = transform(x, H_L)
                right_x = transform(x, H_R)
                diagonal_copy_(left_s, left_x, w)
                diagonal_copy_(right_s, right_x, w)
                diagonal_copy_(normalizer, x_normalizer, w)
            else:
                final_m = transform(x, H)

        final = (final_m + 1e-9).squeeze(1).log() + root
        logZ = final.logsumexp(-1) + x_normalizer.squeeze(-1)

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


class Triton_TDPCFG(PCFG_base):
    def __init__(self):
        super(Triton_TDPCFG, self).__init__()

    def loss(self, rules, lens):
        return self._inside(rules, lens)

    @torch.enable_grad()
    def _inside(
        self, rules, lens, mbr=False, viterbi=False, marginal=False, s_span=None
    ):
        assert viterbi is not True
        unary = rules["unary"]
        root = rules["root"].exp()

        # 3d binary rule probabilities tensor decomposes to three 2d matrices after CP decomposition.
        H = rules["head"]  # (batch, NT, r) r:=rank
        L = rules["left"]  # (batch, NT+T, r)
        R = rules["right"]  # (batch, NT+T, r)

        T = unary.shape[-1]
        S = L.shape[-2]
        NT = S - T
        # r = L.shape[-1]

        L_term = L[:, NT:, ...].contiguous()
        L_nonterm = L[:, :NT, ...].contiguous()
        R_term = R[:, NT:, ...].contiguous()
        R_nonterm = R[:, :NT, ...].contiguous()

        H = H.transpose(-1, -2)
        # (m_A, r_A) + (m_B, r_B) -> (r_A, r_B)
        H_L = torch.matmul(H, L_nonterm)
        H_R = torch.matmul(H, R_nonterm)
        LR = torch.cat([H_L, H_R], dim=-1)

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

        ### Do not remove this, otherwise the gradient would be wrong
        with torch.no_grad():
            unary_max = unary.max(-1)[0]

        unary = (unary - unary_max.unsqueeze(-1)).exp()

        unary = torch.einsum(
            "bnp, bpq -> bnq", unary, torch.cat([L_term, R_term], dim=-1)
        )

        alpha_c = unary.new_zeros(batch, N, N, 2, L.shape[2])
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
            torch.einsum("bnr, br -> b", out, torch.einsum("bm, brm -> br", root, H))
            + 1e-9
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


class MyTDPCFG(TDPCFG):
    def __init__(self):
        super(MyTDPCFG, self).__init__()
        self.training = True

    def train(self, mode):
        self.training = mode

    def _get_span_distribution(
        self, logZ, tocopy_array, span_indicator, lens, allow_grad=True
    ):
        # batch, seq_len = s.shape[:2]
        # ! skip the seq_len>=3 requirement
        # if seq_len >= 3:
        NT = tocopy_array[0].shape[-1]
        B, N = span_indicator.shape[:2]
        assert logZ.requires_grad
        assert not span_indicator.requires_grad
        logZ.sum().backward(
            retain_graph=True, create_graph=allow_grad, inputs=tocopy_array
        )
        grads = torch.autograd.grad(logZ.sum(), tocopy_array, create_graph=allow_grad)
        # print(s.grad)
        # sm_array = [(t.grad / t[:])[:, 0] for t in tocopy_array]
        # print(sm_array[2])
        marginals = span_indicator.new_zeros(B, N, N, NT)  # .requires_grad_()
        for w in range(2, marginals.shape[1]):
            # print(w, tocopy_array[w - 2].grad)
            diagonal_copy_(
                marginals,
                grads[w - 2],
                w,
            )
        # for t in tocopy_array:
        # del t.grad
        tocopy_array = marginals
        marginals = marginals
        # marginals.sum().backward()
        # print([t.grad for t in tocopy_array[:2]])
        # assert torch.all(torch.isclose(marginals, span_indicator.grad))
        return marginals, tocopy_array

    @torch.enable_grad()
    def decode(self, rules, lens, viterbi=False, mbr=False, allow_grad=False):
        return self._inside(
            rules=rules, lens=lens, viterbi=viterbi, mbr=mbr, allow_grad=allow_grad
        )

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
    ):
        assert viterbi is not True
        unary = rules["unary"]
        root = rules["root"]

        # print("unary shape", unary.shape)
        # input()

        unary = unary.masked_fill(
            ~torch.nn.functional.dropout(
                unary.new_ones(unary.shape), p=dropout_pt, training=self.training
            ).bool(),
            -1e9,
        )

        # 3d binary rule probabilities tensor decomposes to three 2d matrices after CP decomposition.
        H = rules["head"]  # (batch, NT, r) r:=rank
        L = rules["left"]  # (batch, NT+T, r)
        R = rules["right"]  # (batch, NT+T, r)

        T = unary.shape[-1]
        S = L.shape[-2]
        NT = S - T
        # r = L.shape[-1]

        L_term = L[:, NT:, ...].contiguous()
        L_nonterm = L[:, :NT, ...].contiguous()
        R_term = R[:, NT:, ...].contiguous()
        R_nonterm = R[:, :NT, ...].contiguous()

        @checkpoint_nonreentrant
        def transform_left_t(x, left):
            """
            :param x: shape (batch, n, T)
            :return: shape (batch, n, r)
            """
            return (x.unsqueeze(-1) + left.unsqueeze(1)).logsumexp(2)

        @checkpoint_nonreentrant
        def transform_left_nt(x, left):
            return (x.unsqueeze(-1) + left.unsqueeze(1)).logsumexp(2)

        @checkpoint_nonreentrant
        def transform_right_t(x, right):
            return (x.unsqueeze(-1) + right.unsqueeze(1)).logsumexp(2)

        @checkpoint_nonreentrant
        def transform_right_nt(x, right):
            return (x.unsqueeze(-1) + right.unsqueeze(1)).logsumexp(2)

        # @checkpoint
        @checkpoint_nonreentrant
        def merge(Y, Z):
            """
            :param Y: shape (batch, n, w, r)
            :param Z: shape (batch, n, w, r)
            :return: shape (batch, n, x)
            """
            # contract dimension w.
            b_n_r = (Y + Z).logsumexp(-2)
            # contract dimension r.
            b_n_x = (b_n_r.unsqueeze(-2) + H.unsqueeze(1)).logsumexp(-1)
            return b_n_x

        batch, N, *_ = unary.shape
        N += 1

        if span_mask is None:
            span_mask = unary.new_ones(batch, N, N, 1).bool()
        if len(span_mask.shape) == 3:
            span_mask = span_mask.unsqueeze(3)

        # for estimating marginals.
        span_indicator = unary.new_zeros(batch, N, N).requires_grad_(mbr)

        left_term = transform_left_t(unary, L_term)
        right_term = transform_right_t(unary, R_term)

        s = unary.new_zeros(batch, N, N, NT).fill_(-1e9)
        # for caching V^{T}s_{i,k} and W^{T}s_{k+1,j} as described in paper to decrease complexities.
        left_s = unary.new_zeros(batch, N, N, L.shape[2]).fill_(-1e9)
        right_s = unary.new_zeros(batch, N, N, L.shape[2]).fill_(-1e9)

        diagonal_copy_(left_s, left_term, w=1)
        diagonal_copy_(right_s, right_term, w=1)

        # w: span width
        tocopy_array = []
        for w in range(2, N):
            # n: the number of spans of width w.
            n = N - w
            Y = stripe(left_s, n, w - 1, (0, 1))
            Z = stripe(right_s, n, w - 1, (1, w), 0)
            x = merge(Y.clone(), Z.clone())
            x = x + span_indicator[:, torch.arange(n), w + torch.arange(n)].unsqueeze(
                -1
            )
            # if dropout > 0.:
            # x += -1e7 * (1-torch.nn.functional.dropout(x.new_ones(x.shape), p = dropout, training = self.training and w<N-3))
            x = x.masked_fill(
                ~torch.nn.functional.dropout(
                    x.new_ones(x.shape), p=dropout, training=self.training
                ).bool(),
                -1e9,
            )
            tocopy = x
            if tocopy.requires_grad:
                tocopy.retain_grad()
            tocopy_array.append(tocopy)
            if w + 1 < N:
                left_x = transform_left_nt(x, L_nonterm)
                right_x = transform_right_nt(x, R_nonterm)
                diagonal_copy_(left_s, left_x, w)
                diagonal_copy_(right_s, right_x, w)
            diagonal_copy_(s, tocopy, w)

            s += ~span_mask * -1e9

        final = s[torch.arange(batch), 0, lens] + root
        logZ = final.logsumexp(-1)

        if mbr or viterbi:
            prediction, marginals = self._get_prediction(
                logZ, span_indicator, lens, mbr=mbr, allow_grad=allow_grad
            )
            return {"prediction": prediction, "partition": logZ}
        elif span_dist:
            marginals, _ = self._get_span_distribution(
                logZ, tocopy_array, span_indicator, lens, allow_grad=allow_grad
            )
            return {
                "partition": logZ,
                "span_marginals": marginals,
                "tocopy_array": tocopy_array,
            }
        else:
            return {"partition": logZ}


class MyTDPCFGFaster(MyTDPCFG):
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
        grads = torch.autograd.grad(
            logZ.sum(),
            tocopy_array,
            create_graph=allow_grad,
            # allow_unused=True,
            # materialize_grads=True,
        )
        corrected_grads = [torch.maximum(g.new_zeros(*g.shape), g) for g in grads]
        # print(s.grad)
        # assert all([torch.all(torch.logical_and(t>0, t<=1)) for t in tocopy_array])
        # grads = [torch.maximum(g.log()-t.log(), g.new_ones(*g.shape)*-1e9).exp() for g, t in zip(grads, tocopy_array)]
        # print(grads[0])
        # if not all([torch.all(torch.logical_and(g>0, g<=1.1)) for g in grads]):
        # print([(g.max(), g.min()) for g in grads])
        # assert all([torch.all(torch.logical_and(g>0, g<=1.1)) for g in grads])
        # sm_array = [(t.grad / t[:])[:, 0] for t in tocopy_array]
        # print(sm_array[2])
        # print([t.shape for t in tocopy_array[:2]])
        marginals = span_indicator.new_zeros(B, N, N, NT)  # .requires_grad_()
        if include_unary:
            diagonal_copy_(marginals, (corrected_grads[0].sum(-1, keepdim=True).expand(-1, -1, NT)/NT), 1)
            corrected_grads = corrected_grads[1:]
        for w in range(2, marginals.shape[1]):
            # print(w, tocopy_array[w - 2].grad)
            diagonal_copy_(
                marginals,
                corrected_grads[w - 2],
                w,
            )
        # for t in tocopy_array:
        # del t.grad
        tocopy_array = marginals
        marginals = marginals
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
        span_dist=False,
        allow_grad=False,
        span_mask=None,
        dropout=0.0,
        dropout_pt=0.0,
        s_span=None,
        include_unary=False,
    ):
        assert viterbi is not True
        unary = rules["unary"].clone()
        root = rules["root"].clone()

        # 3d binary rule probabilities tensor decomposes to three 2d matrices after CP decomposition.
        H = rules["head"].clone()  # (batch, NT, r) r:=rank
        L = rules["left"].clone()  # (batch, NT+T, r)
        R = rules["right"].clone()  # (batch, NT+T, r)

        T = unary.shape[-1]
        S = L.shape[-2]
        NT = S - T
        # r = L.shape[-1]

        L_term = L[:, NT:, ...].contiguous()
        L_nonterm = L[:, :NT, ...].contiguous()
        R_term = R[:, NT:, ...].contiguous()
        R_nonterm = R[:, :NT, ...].contiguous()

        H = H.transpose(-1, -2)
        # (m_A, r_A) + (m_B, r_B) -> (r_A, r_B)
        H_L = torch.matmul(H, L_nonterm)
        H_R = torch.matmul(H, R_nonterm)

        tocopy_array = []
        tocopy_array.append(unary) # first term is unary potentials.

        def transform(x, y):
            """
            :param x: shape (batch, n, T)
            :return: shape (batch, n, r)
            """
            return torch.matmul(x, y)

        # @checkpoint
        @checkpoint_nonreentrant
        def merge(Y, Z, y, z, indicator):
            """
            :param Y: shape (batch, n, w, r)
            :param Z: shape (batch, n, w, r)
            :return: shape (batch, n, x)
            """
            # contract dimension w.
            Y = (Y + 1e-9).log() + y.unsqueeze(-1)
            Z = (Z + 1e-9).log() + z.unsqueeze(-1)
            assert torch.all(~torch.isnan(Y + Z))
            b_n_r = (Y + Z).logsumexp(-2) + indicator
            normalizer = b_n_r.max(-1)[0]
            logbnr = b_n_r
            b_n_r = (b_n_r - normalizer.unsqueeze(-1)).exp()
            # print(b_n_r)
            assert torch.all(~torch.isnan(b_n_r))
            assert torch.all(b_n_r >= 0) and torch.all(b_n_r <= 1)
            return b_n_r, normalizer, logbnr

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

        normalizer = unary.new_zeros(batch, N, N).fill_(-1e9)
        norm = unary.max(-1)[0]
        diagonal_copy_(normalizer, norm, w=1)
        unary = (unary - norm.unsqueeze(-1)).exp()
        left_term = transform(unary, L_term)
        right_term = transform(unary, R_term)

        # for caching V^{T}s_{i,k} and W^{T}s_{k+1,j} as described in paper to decrease complexities.
        left_s = unary.new_zeros(batch, N, N, L.shape[2]).fill_(-1e9)
        right_s = unary.new_zeros(batch, N, N, L.shape[2]).fill_(-1e9)

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
            )
            keep_mask = torch.nn.functional.dropout(
                x.new_ones(x.shape), p=dropout, training=self.training
            ).bool()
            x = x * keep_mask  # $x.masked_fill(dropout_mask, 0)
            # logx = logx #+ ~keep_mask * -1e9  # logx.masked_fill(dropout_mask, -1e9)

            tocopy_array.append(logx)
            if w + 1 < N:
                left_x = transform(x, H_L)
                right_x = transform(x, H_R)
                diagonal_copy_(left_s, left_x, w)
                diagonal_copy_(right_s, right_x, w)
                diagonal_copy_(normalizer, x_normalizer, w)
            else:
                final_m = transform(x, H)

        final = (final_m + 1e-9).squeeze(1).log() + root
        logZ = final.logsumexp(-1) + x_normalizer.squeeze(-1)

        if mbr or viterbi:
            prediction, marginals = self._get_prediction(
                logZ, span_indicator, lens, mbr=mbr, allow_grad=allow_grad
            )
            return {"prediction": prediction, "partition": logZ}
        elif span_dist:
            marginals, _ = self._get_span_distribution(
                logZ, tocopy_array, span_indicator, lens, allow_grad=allow_grad, 
                include_unary=include_unary
            )
            return {
                "partition": logZ,
                "span_marginals": marginals,
                "tocopy_array": tocopy_array,
            }
        else:
            return {"partition": logZ}

        # if not mbr and not viterbi:
        #     return {'partition': logZ}

        # elif marginal:
        #     logZ.sum().backward()
        #     return {'marginal': span_indicator.grad}

        # else:
        #     return {
        #         "prediction": self._get_prediction(logZ, span_indicator, lens, mbr=True),
        #         "partition": logZ
        #     }
