import torch
from parser.pcfgs.fn import stripe, diagonal_copy_, diagonal


class PCFG_base:

    def _inside(self):
        raise NotImplementedError

    def inside(self, rules, lens):
        return self._inside(rules, lens)

    @torch.enable_grad()
    def decode(self, rules, lens, viterbi=False, mbr=False):
        return self._inside(rules=rules, lens=lens, viterbi=viterbi, mbr=mbr)

    def _get_prediction(self, logZ, span_indicator, lens, mbr=False):
        batch, seq_len = span_indicator.shape[:2]
        prediction = [[] for _ in range(batch)]
        # to avoid some trivial corner cases.
        if seq_len >= 3:
            assert logZ.requires_grad
            logZ.sum().backward()
            marginals = span_indicator.grad
            # print(marginals.sum([1, 2]))
            if mbr:
                return self._cky_zero_order(marginals.detach(), lens)
            else:
                viterbi_spans = marginals.nonzero().tolist()
                for span in viterbi_spans:
                    prediction[span[0]].append((span[1], span[2]))
        return prediction

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

    def get_plus_semiring(self, viterbi):
        if viterbi:

            def plus(x, dim):
                return torch.max(x, dim)[0]

        else:

            def plus(x, dim):
                return torch.logsumexp(x, dim)

        return plus

    def _eisner(self, attach, lens):
        self.huge = -1e9
        self.device = attach.device
        """
        :param attach: The marginal probabilities.
        :param lens: sentences lens
        :return: predicted_arcs
        """
        A = 0
        B = 1
        I = 0
        C = 1
        L = 0
        R = 1
        b, N, *_ = attach.shape
        attach.requires_grad_(True)
        alpha = [
            [
                [
                    torch.zeros(b, N, N, device=self.device).fill_(self.huge)
                    for _ in range(2)
                ]
                for _ in range(2)
            ]
            for _ in range(2)
        ]
        alpha[A][C][L][:, :, 0] = 0
        alpha[B][C][L][:, :, -1] = 0
        alpha[A][C][R][:, :, 0] = 0
        alpha[B][C][R][:, :, -1] = 0
        semiring_plus = self.get_plus_semiring(viterbi=True)
        # single root.
        start_idx = 1
        for k in range(1, N - start_idx):
            f = torch.arange(start_idx, N - k), torch.arange(k + start_idx, N)
            ACL = alpha[A][C][L][:, start_idx : N - k, :k]
            ACR = alpha[A][C][R][:, start_idx : N - k, :k]
            BCL = alpha[B][C][L][:, start_idx + k :, N - k :]
            BCR = alpha[B][C][R][:, start_idx + k :, N - k :]
            x = semiring_plus(ACR + BCL, dim=2)
            arcs_l = x + attach[:, f[1], f[0]]
            alpha[A][I][L][:, start_idx : N - k, k] = arcs_l
            alpha[B][I][L][:, k + start_idx : N, N - k - 1] = arcs_l
            x = semiring_plus(ACR + BCL, dim=2)
            arcs_r = x + attach[:, f[0], f[1]]
            alpha[A][I][R][:, start_idx : N - k, k] = arcs_r
            alpha[B][I][R][:, k + start_idx : N, N - k - 1] = arcs_r
            AIR = alpha[A][I][R][:, start_idx : N - k, 1 : k + 1]
            BIL = alpha[B][I][L][:, k + start_idx :, N - k - 1 : N - 1]
            new = semiring_plus(ACL + BIL, dim=2)
            alpha[A][C][L][:, start_idx : N - k, k] = new
            alpha[B][C][L][:, k + start_idx : N, N - k - 1] = new
            new = semiring_plus(AIR + BCR, dim=2)
            alpha[A][C][R][:, start_idx : N - k, k] = new
            alpha[B][C][R][:, start_idx + k : N, N - k - 1] = new
        # dealing with the root.
        root_incomplete_span = alpha[A][C][L][:, 1, : N - 1] + attach[:, 0, 1:]
        for k in range(1, N):
            AIR = root_incomplete_span[:, :k]
            BCR = alpha[B][C][R][:, k, N - k :]
            alpha[A][C][R][:, 0, k] = semiring_plus(AIR + BCR, dim=1)
        logZ = torch.gather(alpha[A][C][R][:, 0, :], -1, lens.unsqueeze(-1))
        arc = torch.autograd.grad(logZ.sum(), attach)[0].nonzero().tolist()
        predicted_arc = [[] for _ in range(logZ.shape[0])]
        for a in arc:
            predicted_arc[a[0]].append((a[1] - 1, a[2] - 1))
        return predicted_arc
    
    def sample_recursive(self, l, r, grad, reward, argmax=False, epsilon=0., span_score = None, gae_gamma=0.95):
        import numpy as np
        # print(l, r)
        assert len(grad.shape) == 2, "grad should be 2d"
        w = r - l
        if w==1: return 0, 0, [(l, r)], 0, [], 0
        if w == 2:
            return 0, reward[l, r], [(l, r)], 0, [(torch.tensor(0.), torch.tensor(0., device=span_score.device), torch.tensor(0.))] if span_score is not None else [], 0# logprob, reward
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
        lp, lr, ls, le, lrl, lgae= self.sample_recursive(l, split, grad, reward, argmax, epsilon=epsilon, span_score=span_score)
        rp, rr, rs, re, rrl, rgae = self.sample_recursive(split, r, grad, reward, argmax, epsilon=epsilon, span_score=span_score)
        # print(lrl, rrl)
        assert split_logp <= 0, f"split logp should be negative, but got {split_logp, p}"
        logprob = split_logp + lp + rp
        rwad = reward[l, r] + lr + rr
        entropy = - (selection_p * log_selection_p).sum() 
        
        if span_score is not None:
            adv = reward[l, r] + span_score[l, split] + span_score[split, r] - span_score[l, r]
            gae = adv + gae_gamma * (lgae+rgae)
        # print("logp, reward", logprob, rwad)
        return logprob, rwad, ls+rs+[(l, r)], entropy, [(split_logp, gae, entropy)] + lrl + rrl if span_score is not None else [], gae if span_score is not None else 0  #lr + rr + reward[l, r]
    
    # def sample_recursive_pcfg(self, prod, l, r, nt, beta, reward, argmax=False, epsilon=0.):
    #     import numpy as np
    #     # print(l, r)
    #     working_beta = beta[:, :, nt]
    #     working_prod = prod[nt, :, :]
    #     assert len(beta.shape) == 3, "grad should be 3d"
    #     w = r - l
    #     if w==1: return working_beta[l, r], 0, [(l, r)], 0
    #     # if w == 2:
    #         # return 0, reward[l, r], [(l, r)], 0 # logprob, reward
    #     p = []
    #     for u in range(l + 1, r):
    #         p.append(working_prod + beta[l, u].unsqueeze(-1) + beta[u, r].unsqueeze(-2))

    #     unnorm_p = torch.torch.stack(p, dim=0)
    #     unnorm_p_shape = unnorm_p.shape
    #     # selection_p = torch.nn.functional.softmax(unnorm_p, 0 )
    #     log_selection_p = torch.nn.functional.log_softmax(unnorm_p.flatten(), 0)        
    #     selection_p = log_selection_p.exp()
    #     # if epsilon > 0:
    #     #     selection_p = selection_p * (1-epsilon) + epsilon/w
    #     #     log_selection_p = log_selection_p + np.log(1-epsilon) + np.log(1/len(p))

    #     if not argmax:
    #         split = torch.multinomial(selection_p , 1).item()
    #     else:
    #         split = selection_p.max(0)[1].item()
    #     split_logp = log_selection_p[split]
    #     SPAN_LEN, NT, _ = unnorm_p_shape 
        
    #     span_split = split // SPAN_LEN
    #     span_split = l + span_split + 1

    #     nt_1, nt_2 = divmod(split % SPAN_LEN) 

    #     lp, lr, ls, le= self.sample_recursive(prod, l, split, nt_1, beta, reward, argmax, epsilon=epsilon)
    #     rp, rr, rs, re = self.sample_recursive(prod, split, r, nt_2, beta, reward, argmax, epsilon=epsilon)
    #     assert split_logp <= 0, f"split logp should be negative, but got {split_logp, p}"
    #     logprob = split_logp + lp + rp
    #     rwad = reward[l, r] + lr + rr
    #     entropy = - (selection_p * log_selection_p).sum() + le + re
    #     # print("logp, reward", logprob, rwad)
    #     return logprob, rwad, ls+rs+[(l, r)], entropy #lr + rr + reward[l, r]
    
    # def sample_pcfg(self, beta, reward, num_samples=1, argmax=False, is_grad_in_log=False, log_normalization = None, epsilon=0., prod =None):
    #     grad = grad.to("cpu")

    #     assert prod is not None, "production rules must be applied"

    #     # if not is_grad_in_log:
    #         # grad = torch.clamp(grad.log(), -30)
        
    #     assert log_normalization is None
    #     assert epsilon == 0, "epsilon is causing worse performance"
    #     # if log_normalization is None:
    #     #     log_norm = torch.zeros(grad.shape[0])
    #     # else:
    #     #     log_norm = log_normalization
    #     reward = reward.to("cpu")
    #     batch, N, _ = reward.shape
    #     N-=1
    #     samples = []
    #     for b in range(batch):
    #         sample = []
    #         for _ in range(num_samples):
    #             # print(f"sampling for {b} batch {_} sample")
    #             out = self.sample_recursive_pcfg(prod, 0, N, beta[b], reward[b], argmax, epsilon=epsilon)
    #             out = list(out)
    #             # out[3] = out[3] - log_norm[b]
    #             sample.append(out)
    #         samples.append(sample)
    #     return samples

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



        root_rule = root_rule.exp()
        # print("root_rule inspection", root_rule)
        # input()


        reward = reward.to("cpu")
        batch, N, _ = reward.shape
        N-=1
        samples = []
        starting_nt = torch.multinomial(root_rule.repeat(1, num_samples).reshape(batch*num_samples, -1), 1).flatten()
        # print("beta, unary shape", precomputed[0].shape, precomputed[1].shape)
        # input()
        nt_idx = 0
        for b in range(batch):
            sample = []
            for _ in range(num_samples):
                # print(f"sampling for {b} batch {_} sample")
                out = self.sample_recursive_pcfg(prod, 0, N, starting_nt[nt_idx], [a[b] for a in precomputed], reward[b], argmax, epsilon=epsilon, tau=tau, span_score=span_score[b] if span_score is not None else None)
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
                # out[3] = out[3] - log_norm[b]
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



    def sample(self, grad, reward, num_samples=1, argmax=False, is_grad_in_log=False, log_normalization = None, epsilon=0., span_score = None):
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
                out = self.sample_recursive(0, N, grad[b], reward[b], argmax, epsilon=epsilon, span_score=span_score[b] if span_score is not None else None)
                out = list(out)
                # print("outsize", len(out))
                if span_score is not None:
                    # print(out[-2])
                    # input()
                    out = out[:-1] # remove the gae return
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

    def build_alpha_chart(self, grad):
        
        batch, N, _ = grad.shape
        grad = grad.unsqueeze(-1)
        alpha = grad.new_zeros(batch, N, N, 1).fill_(0)
        unary = stripe(grad, N-1, 1, (0, 1)).clone()
        if any(unary.flatten() > 0):
            diagonal_copy_(alpha, unary.squeeze(2), 1,)
        # print(alpha.shape, grad.shape, stripe(grad, N-2 , 1, (0, 1)).shape)
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

    def sample_pcfg(self, alpha, reward, num_samples=1, argmax=False, epsilon=0.):
        # grad = torch.clamp(grad.log(), -30)
        alpha = self.build_alpha_chart(grad)
        return self.sample(alpha, reward, num_samples=num_samples, is_grad_in_log=True, epsilon=epsilon), self.sample(alpha, reward, num_samples=1, is_grad_in_log=True, argmax=True), self.entropy(alpha)


        pass

    # def sample_crf(self, grad, reward, num_samples=1, argmax=False, epsilon=0.):
    #     grad = torch.clamp(grad.log(), -30)
    #     alpha = self.build_alpha_chart(grad)
    #     return self.sample(alpha, repward, num_samples=num_samples, is_grad_in_log=True, epsilon=epsilon), self.sample(alpha, reward, num_samples=1, is_grad_in_log=True, argmax=True), self.entropy(alpha)
    def sample_crf(self, grad, reward, num_samples=1, argmax=False, epsilon=0., span_score = None):
        grad = torch.clamp(grad.log(), -30)
        if span_score is None:
            alpha = self.build_alpha_chart(grad)
            return self.sample(alpha, reward, num_samples=num_samples, is_grad_in_log=True, epsilon=epsilon, span_score=span_score), self.sample(alpha, reward, num_samples=1, is_grad_in_log=True, argmax=True, span_score=None), self.entropy(alpha), v_est if span_score is not None else None
        else:
            alpha, v_est, ent = self.build_alpha_chart_a2c(grad, reward, span_score)
            return self.sample(alpha, reward, num_samples=num_samples, is_grad_in_log=True, epsilon=epsilon, span_score=v_est), self.sample(alpha, reward, num_samples=1, is_grad_in_log=True, argmax=True, span_score=None), self.entropy(alpha), v_est if span_score is not None else None

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


