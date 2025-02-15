from sympy import beta
from parsing_by_maxseminfo.parser.pcfgs.pcfgs import PCFG_base
from parsing_by_maxseminfo.parser.pcfgs.fn import (
    stripe,
    diagonal_copy_,
    diagonal,
    checkpoint,
    checkpoint_nonreentrant,
)
import torch


class PCFG(PCFG_base):

    @torch.enable_grad()
    def _inside(self, rules, lens, viterbi=False, mbr=False):
        terms = rules["unary"]
        rule = rules["rule"]
        root = rules["root"]

        batch, N, T = terms.shape
        N += 1
        NT = rule.shape[1]
        S = NT + T

        s = terms.new_zeros(batch, N, N, NT).fill_(-1e9)
        NTs = slice(0, NT)
        Ts = slice(NT, S)

        X_Y_Z = rule[:, :, NTs, NTs].reshape(batch, NT, NT * NT)
        X_y_Z = rule[:, :, Ts, NTs].reshape(batch, NT, NT * T)
        X_Y_z = rule[:, :, NTs, Ts].reshape(batch, NT, NT * T)
        X_y_z = rule[:, :, Ts, Ts].reshape(batch, NT, T * T)

        span_indicator = rule.new_zeros(batch, N, N).requires_grad_(viterbi or mbr)

        def contract(x, dim=-1):
            if viterbi:
                return x.max(dim)[0]
            else:
                return x.logsumexp(dim)

        # nonterminals: X Y Z
        # terminals: x y z
        # XYZ: X->YZ  XYz: X->Yz  ...
        @checkpoint
        def Xyz(y, z, rule):
            n = y.shape[1]
            b_n_yz = (y + z).reshape(batch, n, T * T)
            b_n_x = contract(b_n_yz.unsqueeze(-2) + rule.unsqueeze(1))
            return b_n_x

        @checkpoint
        def XYZ(Y, Z, rule):
            n = Y.shape[1]
            b_n_yz = contract(
                Y[:, :, 1:-1, :].unsqueeze(-1) + Z[:, :, 1:-1, :].unsqueeze(-2), dim=2
            ).reshape(batch, n, -1)
            b_n_x = contract(b_n_yz.unsqueeze(2) + rule.unsqueeze(1))
            return b_n_x

        @checkpoint
        def XYz(Y, z, rule):
            n = Y.shape[1]
            Y = Y[:, :, -1, :, None]
            b_n_yz = (Y + z).reshape(batch, n, NT * T)
            b_n_x = contract(b_n_yz.unsqueeze(-2) + rule.unsqueeze(1))
            return b_n_x

        @checkpoint
        def XyZ(y, Z, rule):
            n = Z.shape[1]
            Z = Z[:, :, 0, None, :]
            b_n_yz = (y + Z).reshape(batch, n, NT * T)
            b_n_x = contract(b_n_yz.unsqueeze(-2) + rule.unsqueeze(1))
            return b_n_x

        for w in range(2, N):
            n = N - w

            Y_term = terms[:, :n, :, None]
            Z_term = terms[:, w - 1 :, None, :]

            if w == 2:
                diagonal_copy_(
                    s,
                    Xyz(Y_term, Z_term, X_y_z)
                    + span_indicator[:, torch.arange(n), torch.arange(n) + w].unsqueeze(
                        -1
                    ),
                    w,
                )
                continue

            n = N - w
            x = terms.new_zeros(3, batch, n, NT).fill_(-1e9)

            Y = stripe(s, n, w - 1, (0, 1)).clone()
            Z = stripe(s, n, w - 1, (1, w), 0).clone()

            if w > 3:
                x[0].copy_(XYZ(Y, Z, X_Y_Z))

            x[1].copy_(XYz(Y, Z_term, X_Y_z))
            x[2].copy_(XyZ(Y_term, Z, X_y_Z))

            diagonal_copy_(
                s,
                contract(x, dim=0)
                + span_indicator[:, torch.arange(n), torch.arange(n) + w].unsqueeze(-1),
                w,
            )

        logZ = contract(s[torch.arange(batch), 0, lens] + root)

        if viterbi or mbr:
            prediction = self._get_prediction(logZ, span_indicator, lens, mbr=mbr)
            return {"partition": logZ, "prediction": prediction}

        else:
            return {"partition": logZ}


class Faster_PCFG(PCFG_base):
    @torch.enable_grad()
    def _inside(self, rules, lens, viterbi=False, mbr=False):
        assert viterbi == False

        terms = rules["unary"]
        rule = rules["rule"]
        root = rules["root"]

        batch, N, T = terms.shape
        N += 1
        NT = rule.shape[1]
        S = NT + T

        s = terms.new_zeros(batch, N, N, NT).fill_(-1e9)
        NTs = slice(0, NT)
        Ts = slice(NT, S)

        rule = rule.exp()
        X_Y_Z = rule[:, :, NTs, NTs].contiguous()
        X_y_Z = rule[:, :, Ts, NTs].contiguous()
        X_Y_z = rule[:, :, NTs, Ts].contiguous()
        X_y_z = rule[:, :, Ts, Ts].contiguous()

        span_indicator = rule.new_zeros(batch, N, N).requires_grad_(viterbi or mbr)

        def contract(x, dim=-1):
            if viterbi:
                return x.max(dim)[0]
            else:
                return x.logsumexp(dim)

        # nonterminals: X Y Z
        # terminals: x y z
        # XYZ: X->YZ
        @checkpoint
        def Xyz(y, z, rule):
            y_normalizer = y.max(-1)[0]
            z_normalizer = z.max(-1)[0]
            y, z = (y - y_normalizer.unsqueeze(-1)).exp(), (
                z - z_normalizer.unsqueeze(-1)
            ).exp()
            x = torch.einsum("bny, bnz, bxyz -> bnx", y, z, rule)
            x = (
                (x + 1e-9).log()
                + y_normalizer.unsqueeze(-1)
                + z_normalizer.unsqueeze(-1)
            )
            return x

        @checkpoint
        def XYZ(Y, Z, rule):
            # n = Y.shape[1]
            Y = Y[:, :, 1:-1, :]
            Z = Z[:, :, 1:-1, :]
            Y_normalizer = Y.max(-1)[0]
            Z_normalizer = Z.max(-1)[0]
            Y, Z = (Y - Y_normalizer.unsqueeze(-1)).exp(), (
                Z - Z_normalizer.unsqueeze(-1)
            ).exp()
            X = torch.einsum("bnwy, bnwz, bxyz -> bnwx", Y, Z, rule)
            X = (
                (X + 1e-9).log()
                + Y_normalizer.unsqueeze(-1)
                + Z_normalizer.unsqueeze(-1)
            )
            X = X.logsumexp(2)
            return X

        @checkpoint
        def XYz(Y, z, rule):
            Y = Y[:, :, -1, :]
            Y_normalizer = Y.max(-1)[0]
            z_normalizer = z.max(-1)[0]
            Y, z = (Y - Y_normalizer.unsqueeze(-1)).exp(), (
                z - z_normalizer.unsqueeze(-1)
            ).exp()
            X = torch.einsum("bny, bnz, bxyz->bnx", Y, z, rule)
            X = (
                (X + 1e-9).log()
                + Y_normalizer.unsqueeze(-1)
                + z_normalizer.unsqueeze(-1)
            )
            return X

        @checkpoint
        def XyZ(y, Z, rule):
            Z = Z[:, :, 0, :]
            y_normalizer = y.max(-1)[0]
            Z_normalizer = Z.max(-1)[0]
            y, Z = (y - y_normalizer.unsqueeze(-1)).exp(), (
                Z - Z_normalizer.unsqueeze(-1)
            ).exp()
            X = torch.einsum("bny, bnz, bxyz-> bnx", y, Z, rule)
            X = (
                (X + 1e-9).log()
                + y_normalizer.unsqueeze(-1)
                + Z_normalizer.unsqueeze(-1)
            )
            return X

        for w in range(2, N):
            n = N - w

            Y_term = terms[
                :,
                :n,
                :,
            ]
            Z_term = terms[:, w - 1 :, :]

            if w == 2:
                diagonal_copy_(
                    s,
                    Xyz(Y_term, Z_term, X_y_z)
                    + span_indicator[:, torch.arange(n), torch.arange(n) + w].unsqueeze(
                        -1
                    ),
                    w,
                )
                continue

            n = N - w
            x = terms.new_zeros(3, batch, n, NT).fill_(-1e9)

            Y = stripe(s, n, w - 1, (0, 1)).clone()
            Z = stripe(s, n, w - 1, (1, w), 0).clone()

            if w > 3:
                x[0].copy_(XYZ(Y, Z, X_Y_Z))

            x[1].copy_(XYz(Y, Z_term, X_Y_z))
            x[2].copy_(XyZ(Y_term, Z, X_y_Z))

            diagonal_copy_(
                s,
                contract(x, dim=0)
                + span_indicator[:, torch.arange(n), torch.arange(n) + w].unsqueeze(-1),
                w,
            )

        logZ = contract(s[torch.arange(batch), 0, lens] + root)

        if viterbi or mbr:
            prediction = self._get_prediction(logZ, span_indicator, lens, mbr=mbr)
            return {"partition": logZ, "prediction": prediction}

        else:
            return {"partition": logZ}


class MyPCFG(PCFG):
    
    def __init__(self):
        super(MyPCFG, self).__init__()
        self.training = True

        self.traced_XYZ = None
    
    def train(self, mode):
        self.training = mode

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
        # logZ.sum().backward(
        #     retain_graph=True, create_graph=allow_grad, inputs=tocopy_array
        # )
        grads = torch.autograd.grad(
            logZ.sum(), tocopy_array, create_graph=allow_grad
        )
        # Delete negative grads as theoretically, negative grads are not possible
        corrected_grads = [torch.maximum(g.new_zeros(*g.shape), g) for g in grads]
        # print(s.grad)
        # sm_array = [(t.grad / t[:])[:, 0] for t in tocopy_array]
        # print(sm_array[2])
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
        # if torch.any(marginals<0):
            # print(grads)
        assert torch.all(marginals >=0), "marginals contains negative values"
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
    def decode(self, rules, lens, viterbi=False, mbr=False, allow_grad=False, span_mask = None):
        return self._inside(
            rules=rules, lens=lens, viterbi=viterbi, mbr=mbr, allow_grad=allow_grad, span_mask=span_mask
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
        span_supervised_mask=None,
        viterbi=False,
        mbr=False,
        span_dist=False,
        allow_grad=False,
        span_mask=None,
        dropout = 0.,
        dropout_pt = 0.,
    ):
        terms = rules["unary"]
        rule = rules["rule"]
        root = rules["root"]

        assert not torch.isnan(terms).any(), "terms contains nan"
        assert not torch.isnan(rule).any(), "rule contains nan"
        assert not torch.isnan(root).any(), "root contains nan"
        
        terms = terms.masked_fill(~torch.nn.functional.dropout(terms.new_ones(terms.shape), p = dropout_pt, training = self.training).bool(), -1e9)



        batch, N, T = terms.shape
        N += 1
        NT = rule.shape[1]
        S = NT + T

        if span_mask is None:
            span_mask = terms.new_ones(batch, N, N, 1).bool()
        if len(span_mask.shape) == 3:
            span_mask = span_mask.unsqueeze(3)

        s = terms.new_zeros(batch, N, N, NT).fill_(-1e9)
        NTs = slice(0, NT)
        Ts = slice(NT, S)

        X_Y_Z = rule[:, :, NTs, NTs].reshape(batch, NT, NT * NT)
        X_y_Z = rule[:, :, Ts, NTs].reshape(batch, NT, NT * T)
        X_Y_z = rule[:, :, NTs, Ts].reshape(batch, NT, NT * T)
        X_y_z = rule[:, :, Ts, Ts].reshape(batch, NT, T * T)

        span_indicator = rule.new_zeros(batch, N, N).requires_grad_(viterbi or mbr)

        def contract(x, dim=-1):
            if viterbi:
                return x.max(dim)[0]
            else:
                return x.logsumexp(dim)

        # nonterminals: X Y Z
        # terminals: x y z
        # XYZ: X->YZ  XYz: X->Yz  ...
        # @checkpoint_nonreentrant
        def Xyz(y, z, rule):
            n = y.shape[1]
            b_n_yz = (y + z).reshape(batch, n, T * T)
            b_n_x = contract(b_n_yz.unsqueeze(-2) + rule.unsqueeze(1))
            return b_n_x

        # @checkpoint_nonreentrant
        def XYZ(Y, Z, rule):
            n = Y.shape[1]
            b_n_yz = contract(
                Y[:, :, 1:-1, :].unsqueeze(-1) + Z[:, :, 1:-1, :].unsqueeze(-2), dim=2
            ).reshape(batch, n, -1)
            b_n_x = contract(b_n_yz.unsqueeze(2) + rule.unsqueeze(1))
            return b_n_x

        # @checkpoint_nonreentrant
        def XYz(Y, z, rule):
            n = Y.shape[1]
            Y = Y[:, :, -1, :, None]
            b_n_yz = (Y + z).reshape(batch, n, NT * T)
            b_n_x = contract(b_n_yz.unsqueeze(-2) + rule.unsqueeze(1))
            return b_n_x

        # @checkpoint_nonreentrant
        def XyZ(y, Z, rule):
            n = Z.shape[1]
            Z = Z[:, :, 0, None, :]
            b_n_yz = (y + Z).reshape(batch, n, NT * T)
            b_n_x = contract(b_n_yz.unsqueeze(-2) + rule.unsqueeze(1))
            return b_n_x

        tocopy_array = []
        for w in range(2, N):
            n = N - w

            Y_term = terms[:, :n, :, None]
            Z_term = terms[:, w - 1 :, None, :]

            if w == 2:
                tocopy = Xyz(Y_term, Z_term, X_y_z) + span_indicator[
                    :, torch.arange(n), torch.arange(n) + w
                ].unsqueeze(-1)
                if tocopy.requires_grad:
                    tocopy.retain_grad()
                diagonal_copy_(
                    s,
                    tocopy,
                    w,
                )
                tocopy_array.append(tocopy)
                continue

            n = N - w
            x = terms.new_zeros(3, batch, n, NT).fill_(-1e9)

            Y = stripe(s, n, w - 1, (0, 1)).clone()
            Z = stripe(s, n, w - 1, (1, w), 0).clone()

            if w > 3:
                x[0].copy_(XYZ(Y, Z, X_Y_Z))

            x[1].copy_(XYz(Y, Z_term, X_Y_z))
            x[2].copy_(XyZ(Y_term, Z, X_y_Z))



            tocopy = contract(x, dim=0) + span_indicator[
                :, torch.arange(n), torch.arange(n) + w
            ].unsqueeze(-1)
            tocopy = tocopy.masked_fill(~torch.nn.functional.dropout(tocopy.new_ones(tocopy.shape), p = dropout, training = self.training).bool(), -1e9)
            if tocopy.requires_grad:
                tocopy.retain_grad()
            tocopy_array.append(tocopy)
            diagonal_copy_(
                s,
                tocopy,
                w,
            )

            s = torch.where(span_mask, s, -1e9)

        logZ = contract(s[torch.arange(batch), 0, lens] + root)

        if viterbi or mbr:
            prediction, marginals = self._get_prediction(
                logZ, span_indicator, lens, mbr=mbr, allow_grad=allow_grad
            )
            return {
                "partition": logZ,
                "prediction": prediction,
                "span_marginals": marginals,
            }

        elif span_dist:
            marginals, _ = self._get_span_distribution(
                logZ, tocopy_array, span_indicator, lens, allow_grad
            )
            return {
                "partition": logZ,
                "span_marginals": marginals,
                "span_indicator": span_indicator,
                "rule_logp": rule,
                'beta': s,
                "tocopy_array": tocopy_array,
            }

        else:
            return {"partition": logZ}



class FasterMyPCFG(MyPCFG):
    
    def sample_pcfg_entrance(self, beta, reward, num_samples=1, argmax=False, is_grad_in_log=False, log_normalization = None, epsilon=0., prod =None, root_rule = None, tau=1., span_score = None):
        # grad = torch.clamp(grad.log(), -30)
        # alpha = self.build_alpha_chart(grad\)
        # beta, reward, num_samples=1, argmax=False, is_grad_in_log=False, log_normalization = None, epsilon=0., prod =None, root_rule = None
        return self.sample_pcfg(beta, reward, num_samples=num_samples, argmax=False, epsilon=0., prod =prod, root_rule = root_rule, tau=tau, span_score=span_score), self.sample_pcfg(beta, reward, num_samples=1, argmax=True, prod =prod, root_rule = root_rule, tau=tau), torch.tensor([0], dtype=torch.float)#, self.entropy(alpha)

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
                # prod_tmp = 
                # print(prod[-1])
                prod_tmp = {k: v[b] for k, v in prod[-1].items() if isinstance(v, torch.Tensor) and k in ["unary", "rule", "root"]}
                out = self.sample_recursive_pcfg([None, None, None, prod_tmp], 0, N, starting_nt[nt_idx], [a[b] for a in precomputed], reward[b], argmax, epsilon=epsilon, tau=tau, span_score=span_score[b] if span_score is not None else None)
                out = list(out)[:-1]
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
                out[3] = (out[3]+ root_rule[b]).logsumexp(0) #- log_norm[b]
                sample.append(out)
            samples.append(sample)
        return samples





    @torch.enable_grad()
    def _inside(       
        self,
        rules,
        lens,
        span_supervised_mask=None,
        viterbi=False,
        mbr=False,
        span_dist=False,
        allow_grad=False,
        span_mask=None,
        dropout = 0.,
        dropout_pt = 0.,
        include_unary=False,
        reward = None,
        ):

        assert viterbi == False
        c_vest = reward is not None

        terms = rules["unary"]
        rule = rules["rule"]
        root = rules["root"]

        batch, N, T = terms.shape
        N += 1
        NT = rule.shape[1]
        S = NT + T

        if span_mask is not None:
            assert span_mask.shape == (batch, N, N) 
            span_mask = span_mask.unsqueeze(-1)
            additive_span_mask = (1-span_mask) * -1e9
            # print("span_mask", span_mask[0, :, :, 0])
            # print("additive_mask", additive_span_mask[0, :, :, 0])

        s = terms.new_zeros(batch, N, N, NT).fill_(-1e9)
        if c_vest:
            v_est = terms.new_zeros(batch, N, N, NT).fill_(1e-9)
        else:
            v_est = None
        NTs = slice(0, NT)
        Ts = slice(NT, S)

        rule = rule.exp()
        X_Y_Z = rule[:, :, NTs, NTs].contiguous()
        X_y_Z = rule[:, :, Ts, NTs].contiguous()
        X_Y_z = rule[:, :, NTs, Ts].contiguous()
        X_y_z = rule[:, :, Ts, Ts].contiguous()

        span_indicator = rule.new_zeros(batch, N, N).requires_grad_(viterbi or mbr)

        def contract(x, dim=-1):
            if viterbi:
                return x.max(dim)[0]
            else:
                return x.logsumexp(dim)

        # nonterminals: X Y Z
        # terminals: x y z
        # XYZ: X->YZ
        # @checkpoint_nonreentrant
        def Xyz(y, z, rule):
            y_normalizer = y.max(-1)[0]
            z_normalizer = z.max(-1)[0]
            y, z = (y - y_normalizer.unsqueeze(-1)).exp(), (
                z - z_normalizer.unsqueeze(-1)
            ).exp()
            x = torch.einsum("bny, bnz, bxyz -> bnx", y, z, rule)
            x = (
                (x + 1e-9).log()
                + y_normalizer.unsqueeze(-1)
                + z_normalizer.unsqueeze(-1)
            )
            return x
        
        # @torch.compile
        # @checkpoint_nonreentrant
        def XYZ(Y, Z, rule):
            # n = Y.shape[1]
            Y = Y[:, :, 1:-1, :]
            Z = Z[:, :, 1:-1, :]
            Y_normalizer = Y.max(-1)[0]
            Z_normalizer = Z.max(-1)[0]
            Y, Z = (Y - Y_normalizer.unsqueeze(-1)).exp(), (
                Z - Z_normalizer.unsqueeze(-1)
            ).exp()
            X = torch.einsum("bnwy, bnwz, bxyz -> bnwx", Y, Z, rule)
            X = (
                (X + 1e-9).log()
                + Y_normalizer.unsqueeze(-1)
                + Z_normalizer.unsqueeze(-1)
            )
            X = X.logsumexp(2)
            return X
        
        def XYZ_Vacc(Y, Z, rule, VY, VZ, RX):
            # print("XYZ_Vacc shapes:", "Y", Y.shape, "Z",  Z.shape, "VY", VY.shape, "VZ", VZ.shape, "RX", RX.shape)
            # n = Y.shape[1]
            Y = Y[:, :, 1:-1, :]
            Z = Z[:, :, 1:-1, :]
            VY = VY[:, :, 1:-1, :]
            VZ = VZ[:, :, 1:-1, :]
            Y_normalizer = Y.max(-1)[0]
            Z_normalizer = Z.max(-1)[0]
            Y, Z = (Y - Y_normalizer.unsqueeze(-1)).exp(), (
                Z - Z_normalizer.unsqueeze(-1)
            ).exp()
            VX_1 = torch.einsum("bnwy, bnwz, bxyz -> bnwx", Y*VY, Z, rule)
            VX_2 = torch.einsum("bnwy, bnwz, bxyz -> bnwx", Y, Z*VZ, rule)
            VX = torch.stack([VX_1, VX_2], dim = -1).sum(-1)
            # print("VX in XYZ", VX)
            VX = (
                (VX + 1e-9).log()
                + Y_normalizer.unsqueeze(-1)
                + Z_normalizer.unsqueeze(-1)
            )
            VX = VX.logsumexp(2)#.exp()
            VX = VX#+RX.unsqueeze(-1)
            
            return VX


        # @checkpoint_nonreentrant
        def XYz(Y, z, rule):
            Y = Y[:, :, -1, :]
            Y_normalizer = Y.max(-1)[0]
            z_normalizer = z.max(-1)[0]
            Y, z = (Y - Y_normalizer.unsqueeze(-1)).exp(), (
                z - z_normalizer.unsqueeze(-1)
            ).exp()
            X = torch.einsum("bny, bnz, bxyz->bnx", Y, z, rule)
            X = (
                (X + 1e-9).log()
                + Y_normalizer.unsqueeze(-1)
                + z_normalizer.unsqueeze(-1)
            )
            return X
        
        # @checkpoint_nonreentrant
        def XYz_Vacc(Y, z, rule, VY, RX):
            # print("XYz_Vacc shapes:", "Y", Y.shape, "z", z.shape, "VY", VY.shape, "RX", RX.shape)
            # print("XYz_Vacc shapes:", Y.shape, z.shape, VY.shape, RX.shape)
            Y = Y[:, :, -1, :]
            VY = VY[:, :, -1, :]
            Y_normalizer = Y.max(-1)[0]
            z_normalizer = z.max(-1)[0]
            Y, z = (Y - Y_normalizer.unsqueeze(-1)).exp(), (
                z - z_normalizer.unsqueeze(-1)
            ).exp()
            VX = torch.einsum("bny, bnz, bxyz->bnx", Y*VY, z, rule)
            VX = (
                (VX + 1e-9).log()
                + Y_normalizer.unsqueeze(-1)
                + z_normalizer.unsqueeze(-1)
            )
            VX = VX#.exp() + RX.unsqueeze(-1)
            return VX

        # @checkpoint_nonreentrant
        def XyZ(y, Z, rule):
            Z = Z[:, :, 0, :]
            
            y_normalizer = y.max(-1)[0]
            Z_normalizer = Z.max(-1)[0]
            y, Z = (y - y_normalizer.unsqueeze(-1)).exp(), (
                Z - Z_normalizer.unsqueeze(-1)
            ).exp()
            X = torch.einsum("bny, bnz, bxyz-> bnx", y, Z, rule)
            X = (
                (X + 1e-9).log()
                + y_normalizer.unsqueeze(-1)
                + Z_normalizer.unsqueeze(-1)
            )
            return X
        
        # @checkpoint_nonreentrant
        def XyZ_Vacc(y, Z, rule, VZ, RX):
            # print("XyZ_Vacc shapes:", "y", y.shape, "Z", Z.shape, "VZ", VZ.shape, "RX", RX.shape)
            # print("XyZ_Vacc shapes:", y.shape, Z.shape, VZ.shape, RX.shape)
            Z = Z[:, :, 0, :]
            VZ = VZ[:, :, 0, :]
            y_normalizer = y.max(-1)[0]
            Z_normalizer = Z.max(-1)[0]
            y, Z = (y - y_normalizer.unsqueeze(-1)).exp(), (
                Z - Z_normalizer.unsqueeze(-1)
            ).exp()
            VX = torch.einsum("bny, bnz, bxyz-> bnx", y, Z*VZ, rule)
            VX = (
                (VX + 1e-9).log()
                + y_normalizer.unsqueeze(-1)
                + Z_normalizer.unsqueeze(-1)
            )
            VX = VX#.exp() + RX.unsqueeze(-1)
            return VX

        # tocopy_array = []
        tocopy_array = []
        tocopy_array.append(terms) # first term is unary potentials.
        # print("terms", rules)

        for w in range(2, N):
            n = N - w

            Y_term = terms[
                :,
                :n,
                :,
            ]
            Z_term = terms[:, w - 1 :, :]

            if w == 2:
                tocopy = Xyz(Y_term, Z_term, X_y_z) + span_indicator[
                    :, torch.arange(n), torch.arange(n) + w
                ].unsqueeze(-1)
                diagonal_copy_(
                    s,
                    tocopy,
                    w,
                )
                tocopy_array.append(tocopy)
                if c_vest:
                    rw = diagonal(reward, w)

                    # v_values = Xyz_Vacc(Y_term, Z_term, X_y_z, tocopy_array[-1])
                    # print("reward", rw.shape, v_est.shape, w)
                    diagonal_copy_(
                        v_est,
                        rw.unsqueeze(-1).repeat(1, 1, NT),
                        w,
                    )
                continue

            n = N - w
            x = terms.new_zeros(3, batch, n, NT).fill_(-1e9)
            vx = terms.new_zeros(3, batch, n, NT).fill_(-1e9)

            Y = stripe(s, n, w - 1, (0, 1)).clone()
            Z = stripe(s, n, w - 1, (1, w), 0).clone()
            if c_vest:
                RX = diagonal(reward, w)
                VY = stripe(v_est, n, w - 1, (0, 1)).clone()
                VZ = stripe(v_est, n, w - 1, (1, w), 0).clone()

            if w > 3:
                x[0].copy_(XYZ(Y, Z, X_Y_Z))
                if c_vest:
                    vx[0].copy_(XYZ_Vacc(Y, Z, X_Y_Z, VY, VZ, RX))



            x[1].copy_(XYz(Y, Z_term, X_Y_z))
            x[2].copy_(XyZ(Y_term, Z, X_y_Z))

            if c_vest:
                vx[1].copy_(XYz_Vacc(Y, Z_term, X_Y_z, VY, RX))
                vx[2].copy_(XyZ_Vacc(Y_term, Z, X_y_Z, VZ, RX))

            tocopy = contract(x, dim=0) + span_indicator[
                :, torch.arange(n), torch.arange(n) + w
            ].unsqueeze(-1)



            tocopy_array.append(tocopy)
            # print("tocopy", tocopy.max())
            diagonal_copy_(
                s,
                tocopy,
                w,
            )
            if span_mask is not None:
                s = (s + additive_span_mask).clamp(-1e9)
            if c_vest:
                # print("premerge vx RX.shape", vx.shape, RX.shape, tocopy.shape)
                # print("pre")
                vx = (vx.logsumexp(0) - tocopy).exp() + RX[ :, :, None]
                # print("postmerge vx", vx[:, 0])
                diagonal_copy_(
                    v_est,
                    vx,
                    w,
                )

        logZ = contract(s[torch.arange(batch), 0, lens] + root)
        if c_vest:
            v0 = (v_est[torch.arange(batch), 0, lens] * root.exp()).sum(-1)

        # print(logZ)

        if viterbi or mbr:
            prediction, marginals = self._get_prediction(
                logZ, span_indicator, lens, mbr=mbr, allow_grad=allow_grad
            )
            return {
                "partition": logZ,
                "prediction": prediction,
                "span_marginals": marginals,
            }

        elif span_dist:
            marginals, _ = self._get_span_distribution(
                logZ, tocopy_array, span_indicator, lens, allow_grad=allow_grad, 
                include_unary = include_unary
            )
            return {
                "partition": logZ,
                "span_marginals": marginals,
                "span_indicator": span_indicator,
                "rule_logp": rule,
                # 'beta': s,
                "tocopy_array": tocopy_array,
                "v_est": v_est,
            }

        else:
            return {"partition": logZ, "v_est": v_est, "beta": s, "v0": v0}



    def sample_recursive_pcfg(self, prod_tuple, l, r, nt, precomputed, reward, argmax=False, epsilon=0., LR=False, tau = 1., span_score = None, v_est = None, gae_gamma=0.95):
        # LR = False -> left LR = True -> right
        import numpy as np

        beta, unary = precomputed
        # print("beta", beta[:, :, 0])
        _, _, _, rules = prod_tuple
        
        terms = rules["unary"]
        rule = rules["rule"]
        root = rules["root"]
        assert len(terms.shape) == 2, f"terms should have a shape of (l, NT)"

        w = r - l
        if w==1: 
            # print(unary[l].max(0))
            # v_est[l, r] = 0
            return torch.tensor(0, device = beta.device), 0, [(l, r)], terms[l], [], 0
        if w == 2:
            if w == len(beta)-1:
                raise NotImplementedError("this line should not be touched")
            return torch.tensor(0, device = beta.device), reward[l, r], [(l, r)], beta[l, r], [(torch.tensor(0., device=span_score.device), torch.tensor(0., device=span_score.device), torch.tensor(0., device=span_score.device))] if span_score is not None else [], 0 
        
        # print(terms, rule, root)

        # print(rules)
        N, T = terms.shape
        N += 1
        NT = beta.shape[-1]
        S = NT + T

        NTs = slice(0, NT)
        Ts = slice(NT, S)
        PT = S-NT

        rule = rule.exp()
        # print("rule counts", NT, N, T)
        # print("general rule shape", rule.shape)
        # print("span state", l, r, nt)
        X_Y_Z = rule[nt, NTs, NTs].contiguous()
        X_y_Z = rule[nt, Ts, NTs].contiguous()
        X_Y_z = rule[nt, NTs, Ts].contiguous()
        X_Y_Z_complete = rule[:, NTs, NTs].contiguous()
        X_y_Z_complete = rule[:, Ts, NTs].contiguous()
        X_Y_z_complete = rule[:, NTs, Ts].contiguous()
        # print("specific rule shape", X_Y_Z.shape, X_y_Z.shape, X_Y_z.shape)
        # X_y_z = rule[:, nt, Ts, Ts].contiguous()


        def XYZ(Y, Z, rule):
            # Y: (w, NT)
            # rule: (1, NT, NT)
            # n = Y.shape[1]
            # Y = Y[:, :, :]
            # Z = Z[:, :, 1:-1, :]
            if len(rule.shape) == 2: rule = rule.unsqueeze(0)
            # print("XYZ shape", "Y", Y.shape, "Z", Z.shape, "rule", rule.shape)
            Y_normalizer = Y.max(-1)[0]
            Z_normalizer = Z.max(-1)[0]
            # print(Y_normalizer, Z_normalizer)
            Y, Z = (Y - Y_normalizer.unsqueeze(-1)).exp(), (
                Z - Z_normalizer.unsqueeze(-1)
            ).exp()
            # print(Y)
            X = torch.einsum("wy, wz, xyz -> wx", Y, Z, rule)
            # print(X)
            X = (
                (X + 1e-9).log()
                + Y_normalizer.unsqueeze(-1)
                + Z_normalizer.unsqueeze(-1)
            )
            # print("XYZ shape", "X", X.shape)
            # print("X out", X)
            # X = X.logsumexp(2)
            return X.squeeze(1)
        


        # @checkpoint_nonreentrant
        def XYz(Y, z, rule):
            # Y: (PT)
            # Z: (NT)
            # rule: (1, PT, NT)
            # Y = Y[:, :, -1, :]
            if len(rule.shape) == 2: rule = rule.unsqueeze(0)
            Y_normalizer = Y.max(-1)[0]
            z_normalizer = z.max(-1)[0]
            Y, z = (Y - Y_normalizer.unsqueeze(-1)).exp(), (
                z - z_normalizer.unsqueeze(-1)
            ).exp()
            X = torch.einsum("y, z, xyz->x", Y, z, rule)
            X = (
                (X + 1e-9).log()
                + Y_normalizer.unsqueeze(-1)
                + z_normalizer.unsqueeze(-1)
            )
            return X
        

        # @checkpoint_nonreentrant
        def XyZ(y, Z, rule):
            if len(rule.shape) == 2: rule = rule.unsqueeze(0)
            Z_normalizer = Z.max(-1)[0]
            y_normalizer = y.max(-1)[0]
            y, Z = (y - y_normalizer.unsqueeze(-1)).exp(), (
                Z - Z_normalizer.unsqueeze(-1)
            ).exp()
            X = torch.einsum("y, z, xyz->x", y, Z, rule)
            X = (
                (X + 1e-9).log()
                + y_normalizer.unsqueeze(-1)
                + Z_normalizer.unsqueeze(-1)
            )
            return X

        p = []
        
        assert len(beta.shape) == 3, "grad should be 3d"

        if w>3:
            Ys = torch.stack([beta[l, u] for u in range(l+2, r-1)], dim=0) # (w, NT)
            assert torch.all(Ys>-1000), f"Ys should be larger than -1000, {Ys}"
            Zs = torch.stack([beta[u, r] for u in range(l+2, r-1)], dim=0) # (w, NT)
            p.append(XYZ(Ys, Zs, X_Y_Z))
        p = [XyZ(unary[l], beta[l+1, r], X_y_Z)] + p
        p.append(XYz(beta[l, r-1], unary[r-1], X_Y_z))
        # input()

        # for u in range(l + 2, r-1):

        #     p.append(XYZ(beta[l, u], beta[u, r], X_Y_Z))
        #     p.append(matmul_kernel(L_m[:, nt], beta[l, u]) + matmul_kernel(R_m[:, nt], beta[u, r]))
        # p = [matmul_kernel(L_p[:, nt], unary[l]) + matmul_kernel(R_m[:, nt], beta[l+1, r])] + p
        # p.append(matmul_kernel(R_p[:, nt], unary[r-1]) + matmul_kernel(L_m[:, nt], beta[l, r-1]))
                
                # L_m[:, nt].unsqueeze(-1).log() + R_m[:, nt].unsqueeze(-2).log() + beta[l, u].unsqueeze(-1) + beta[u, r].unsqueeze(-2))
        # p = [L_p[:, nt].unsqueeze(-1).log() + R_m[:, nt].unsqueeze(-2).log() + unary[l].unsqueeze(-1) + beta[l+1, r].unsqueeze(-2)]+p
        # p.append(L_m[:, nt].unsqueeze(-1).log() + R_p[:, nt].unsqueeze(-2).log() + beta[l, r-1].unsqueeze(-1) + unary[r-1].unsqueeze(-2))

        unnorm_p_span = torch.cat(p, dim=0)#[it.flatten().squeeze(0) for it in p]
        # assert torch.isclose(unnorm_p_span.logsumexp(0), beta[l, r, nt]), f"unnorm_p_span must be equal to beta {unnorm_p_span.logsumexp(0), beta[l, r, nt], beta[l, r]}"
        # print(unnorm_p_span)
        assert len(unnorm_p_span)>0, f"the span_selection must have more than one choices"
        # print(unnorm_p_span)
        # unnorm_p_span = torch.stack(unnorm_p_span, dim=0)
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
                nt_logits = X_y_Z.log() + beta[l+1, r].unsqueeze(-2) + unary[l].unsqueeze(-1)
                # nt_logits = L_p[:, nt].unsqueeze(-1).log() + R_m[:, nt].unsqueeze(-2).log() + unary[l].unsqueeze(-1) + beta[l+1, r].unsqueeze(-2) 
            elif span_split == r-1:
                nt_logits = X_Y_z.log() + beta[l, r-1].unsqueeze(-1) + unary[r-1].unsqueeze(-2)
                # nt_logits = L_m[:, nt].unsqueeze(-1).log() + R_p[:, nt].unsqueeze(-2).log() + beta[l, r-1].unsqueeze(-1) + unary[r-1].unsqueeze(-2) 
            else:
                nt_logits = X_Y_Z.log() + beta[l, span_split].unsqueeze(-1) + beta[span_split, r].unsqueeze(-2)
                # nt_logits = L_m[:, nt].unsqueeze(-1).log() + R_m[:, nt].unsqueeze(-2).log() + beta[l, span_split].unsqueeze(-1) + beta[span_split, r].unsqueeze(-2)
            nt_selection_logp = torch.log_softmax(nt_logits.flatten()/tau, 0)
            nt_choice = torch.multinomial(nt_selection_logp.exp(), 1).item()
            # action_logp = action_logp #+ nt_selection_logp[nt_choice]
        else:
            span_split = selection_logp.max(0)[1].item()
            action_logp = selection_logp[span_split]

            span_split += l +1
            
            if span_split == l+1:
                nt_logits = X_y_Z.log() + beta[l+1, r].unsqueeze(-2) + unary[l].unsqueeze(-1)
                # nt_logits = L_p[:, nt].unsqueeze(-1).log() + R_m[:, nt].unsqueeze(-2).log() + unary[l].unsqueeze(-1) + beta[l+1, r].unsqueeze(-2) 
            elif span_split == r-1:
                nt_logits = X_Y_z.log() + beta[l, r-1].unsqueeze(-1) + unary[r-1].unsqueeze(-2)
                # nt_logits = L_m[:, nt].unsqueeze(-1).log() + R_p[:, nt].unsqueeze(-2).log() + beta[l, r-1].unsqueeze(-1) + unary[r-1].unsqueeze(-2) 
            else:
                nt_logits = X_Y_Z.log() + beta[l, span_split].unsqueeze(-1) + beta[span_split, r].unsqueeze(-2)
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


        logprob = action_logp + lp + rp
        rwad = reward[l, r] + lr + rr
        # rwad = reward[l, r] + lr + rr
        entropy = torch.tensor(0., device=beta.device) #- (selection_p * log_selection_p).sum() + le + re
        # print("logp, reward", logprob, rwad)
        # return logprob, rwad, ls+rs+[(l, r)], None #lr + rr + reward[l, r]
        
        # if w == len(beta)-1:

        if span_split == l+1:
            # print(l, span_split, r, le.shape, re.shape)
            converted_logp = XyZ(le, re, X_y_Z_complete)
            # assert nt_1<PT and nt_2<NT, f"nt1 - PT, nt2-NT {nt_1, nt_2}"
        elif span_split == r-1:
            # print(l, span_split, r, le.shape, re.shape)
            converted_logp = XYz(le, re, X_Y_z_complete)
            # nt_1, nt_2 = divmod(nt_choice, PT)
            # assert nt_1<NT and nt_2<PT, f"nt1 - NT, nt2-PT {nt_1, nt_2}"
        else:
            # print(l, span_split, r, le.shape, re.shape)
            # converted_logp = le[:, None] + re[None, :]
            converted_logp = XYZ(le[None, :], re[None, :], X_Y_Z_complete).squeeze(0)
        # converted_logp = converted_logp.exp()
        assert len(converted_logp.shape) == 1, f"inside logp should only have shape of (NT), {converted_logp.shape}"
            # nt_1, nt_2 = divmod(nt_choice, NT)
            # assert nt_1<NT and nt_2<NT, f"nt1, 2 must both be NTs {nt_1, nt_2}"

        if span_score is not None:
            # print("shapes", reward.shape, span_score.shape, l, r, span_split)
            l_ss = span_score[l, span_split,nt_1] if span_split-l>1 else 0
            r_ss = span_score[span_split, r, nt_2] if r-span_split>1 else 0

            adv = reward[l, r] + l_ss + r_ss - span_score[l, r, nt]#.mean(-1)
            gae = adv + gae_gamma * (lgae+rgae)
            

        # print(f"({l, r}): {lrl, rrl}")
        # print("logp, reward", logprob, rwad)
        return logprob, rwad, ls+rs+[(l, r)], converted_logp, [(action_logp + nt_selection_logp[nt_choice], gae, entropy)] + lrl + rrl if span_score is not None else [], gae if span_score is not None else 0 #lr + rr + reward[l, r]