import torch
import numpy as np

from parsing_by_maxseminfo.parser.helper.utils_spanoverlap import removing_punctuations



def compute_span_distance_mtx(form, pas):
    seq_len = len(form)
    # print(form)
    # print([((i, j), ' '.join(form[i:j])) for i in range(seq_len) for j in range(i+1, seq_len+1)])
    span_idx, span_form = zip(
        *[
            ((i, j), " ".join(form[i:j]))
            for i in range(seq_len)
            for j in range(i + 1, seq_len + 1)
        ]
    )
    span_repr = bge_model.encode(span_form)["dense_vecs"]

    pas_span_idx, pas_span_form = zip(
        *[
            ((exi, i, j), " ".join(pas[exi][i:j]))
            for exi in range(len(pas))
            for i in range(len(pas[exi]))
            for j in range(i + 1, len(pas[exi]) + 1)
        ]
    )
    span_passpan_sim_mask = torch.tensor(
        [
            [
                (
                    True
                    if abs((i[1] - i[0]) - (j[2] - j[1])) <= 3 and i[1] - i[0] <= 4
                    else False
                )
                for j in pas_span_idx
            ]
            for i in span_idx
        ]
    )
    split_points = [0]
    current_i = 0
    for i, pas_span_i in enumerate(pas_span_idx):
        if pas_span_i[0] > current_i:
            split_points.append(i - sum(split_points))
            current_i = pas_span_i[0]
    split_points = split_points[1:] + [len(pas_span_idx) - sum(split_points)]

    pas_span_repr = bge_model.encode(pas_span_form, batch_size=128)["dense_vecs"]
    span_passpan_sim = torch.tensor(span_repr @ pas_span_repr.T) * span_passpan_sim_mask
    span_passpan_sim = span_passpan_sim.split(split_points, dim=1)
    span_score = torch.stack([t.max(1).values for t in span_passpan_sim], dim=1).mean(1)
    span_score_mtx = torch.zeros(seq_len + 1, seq_len + 1)
    # print(span_score_mtx.shape)
    for i, score in zip(span_idx, span_score):
        # print(i, score, span_score_mtx[i[0], i[1]])
        span_score_mtx[i[0], i[1]] = score

    return span_score_mtx


def compute_similarity(x, y):
    # return torch.clamp(1 - (x @ y.T), min=1e-9, max=1)
    return -torch.log(torch.clamp(x @ y.T, min=1e-9, max=1))


def compute_similarity_euclidean(x, y):
    # print(torch.norm(x.unsqueeze(1) - y.unsqueeze(0), dim=2).shape)
    return torch.norm(x.unsqueeze(1) - y.unsqueeze(0), dim=2)


def compute_similarity_euclidean_exp(x, y):
    # print(torch.norm(x.unsqueeze(1) - y.unsqueeze(0), dim=2).shape)
    return torch.norm(x.unsqueeze(1) - y.unsqueeze(0), dim=2).exp()


def compute_similarity_euclidean_01(x, y):
    # print(torch.norm(x.unsqueeze(1) - y.unsqueeze(0), dim=2).shape)
    return torch.where(torch.norm(x.unsqueeze(1) - y.unsqueeze(0), dim=2) < 0.2, 0, 100)



def compute_sentence_logprob(
    paraphrased, maskout_length, model, tokenizer, temperature=1
):
    num_samples = paraphrased.input_ids.shape[0]
    logprob_mask = torch.arange(0, paraphrased.input_ids.shape[1]).unsqueeze(0).repeat(
        num_samples, 1
    ) >= maskout_length.unsqueeze(1)
    # print(tokenizer.batch_decode(paraphrased.input_ids))
    # print(maskout_length)
    # print(logprob_mask)
    with torch.no_grad():
        logprobs = torch.log_softmax(
            model(**paraphrased.to("cuda")).logits / temperature, dim=2
        )

    token_logprobs = (
        logprobs[:, :-1]
        .gather(2, paraphrased.input_ids[:, 1:].unsqueeze(-1))
        .squeeze(2)
        .cpu()
    )

    # for logp, token, mask in zip(token_logprobs[0], tokenizer.convert_ids_to_tokens(paraphrased.input_ids[0])[1:], logprob_mask[0, 1:]):
    # print(logp, token, mask)

    return (token_logprobs * logprob_mask[:, 1:]).sum(1)


# from ipynb import prep

# from ipynb_notebooks.utils import SparkLemmatizer
# from parsing_by_maxseminfo.utils.utils import SpacyLemmatizer as SL


def normalizing_string(strg, stemmer):
    # return normalizing_string_by_lemma_he(strg, stemmer)
    # print(stemmer, isinstance(stemmer, SL), type(stemmer))
    if isinstance(stemmer, SL):
        return normalizing_string_by_lemma(strg, stemmer)
    else:
        assert not isinstance(stemmer, SL)
        return normalizing_string_by_stemmer(strg, stemmer)


def normalizing_string_by_lemma(strg, stemmer):
    strg = strg.translate(str.maketrans("", "", string.punctuation)).lower()
    strg = stemmer.lemmatize(strg)
    return strg


def normalizing_string_by_stemmer(strg, stemmer):
    strg = strg.translate(str.maketrans("", "", string.punctuation)).lower()
    strg = " ".join(
        [
            stemmer.stemWord(w).strip() if stemmer.stemWord(w) is not None else w
            for w in word_tokenize(strg)
        ]
    )
    return strg


def cyk(score, count_unigram=False, rb_bias=True, subtree_discount=1.0):
    length = score.shape[0]
    chart = np.zeros_like(score)
    if count_unigram:
        for i in range(length - 1):
            chart[i, i + 1] = score[i, i + 1]
    split_tracker = {}
    for d in range(2, length):
        for i in range(length - d):
            # print(i, d)
            candidates = [chart[i, k] + chart[k, i + d] for k in range(i + 1, i + d)]
            if rb_bias:
                max_split = np.argmax(candidates)
            else:
                winner = np.argwhere(candidates == np.amax(candidates))
                max_split = np.random.choice(winner.reshape(-1), 1)[0]
            # print(max_split)
            chart[i, i + d] = score[i, i + d] + subtree_discount * candidates[max_split]
            split_tracker[(i, i + d)] = max_split + i + 1

    def backtrack(node):
        # print(node)
        if node not in split_tracker.keys():
            return []
        else:
            split = split_tracker[node]
            return [node] + backtrack((node[0], split)) + backtrack((split, node[1]))

    return chart, split_tracker, backtrack((0, length - 1))


# import tkinter
# from turtle import right
# from numpy import indices

import random


def get_normalized_span_prob(
    span_marginals, max_bandwidth, flag_normalize_prob=True, flag_log_mode=False
):
    device = span_marginals.device
    # print(span_marginals.shape)
    # print(span_marginals)
    # masklen = len(span_marginals)
    # mask_1 = torch.arange(masklen, device = device).float()[None, :].expand(masklen, masklen)
    # mask_2 = mask_1.T
    # mask = (mask_2 - mask_1).abs()<=max_bandwidth
    # print(span_marginals)
    if flag_normalize_prob:
        # print(span_marginals.sum())
        if not torch.all(span_marginals < 1.1) or not torch.all(span_marginals >= 0):
            print(span_marginals)
        assert torch.all(span_marginals < 1.1)
        assert torch.all(span_marginals >= 0)
        assert torch.isclose(
            torch.sum(span_marginals), span_marginals.new_ones(1), atol=1e-2
        ), f"current sum is {torch.sum(span_marginals)}"
    indices = (span_marginals).nonzero(as_tuple=True)
    # print(span_marginals)

    nonzero_marginals = span_marginals[indices]

    # print(span_marginals.shape, nonzero_marginals.shape)
    # print(indices)
    # print(nonzero_marginals)
    if flag_normalize_prob:
        if not flag_log_mode:
            return nonzero_marginals / nonzero_marginals.sum(), indices
        else:
            nonzero_marginals = torch.maximum(
                nonzero_marginals.log(),
                nonzero_marginals.new_ones(*nonzero_marginals.shape) * -1e9,
            )  # (nonzero_marginals+1e-37).log()
            return (
                nonzero_marginals
                - torch.logsumexp(nonzero_marginals, dim=-1, keepdim=True),
                indices,
            )
        # return nonzero_marginals / nonzero_marginals.sum(), indices
    else:
        return (nonzero_marginals+1e-9).log(), indices
        # return (
        #     torch.maximum(
        #         nonzero_marginals.log(),
        #         nonzero_marginals.new_ones(*nonzero_marginals.shape) * -1e9,
        #     ),
        #     indices,
        # )



def compute_oneside_matching_with_spanoverlap_score(
    target_spanoverlap_score_array,
    pas_spanoverlap_score_array,
    pas_len_array,
    target_span_marginals_array,
    pas_span_marginals_array,
    pasid2groupid_map,
    flag_normalize_marginals,
    spanoverlap_mask_array=None,
    max_bandwidth=1000,
    return_negative_score=False,
    match_character_only=False,
):
    from ipynb.spanoverlap import SpanScorer

    scorer = SpanScorer()
    from collections import Counter
    import itertools

    if spanoverlap_mask_array is None:
        spanoverlap_mask_array = [
            None for _ in range(len(target_spanoverlap_score_array))
        ]
    batch_size = len(target_spanoverlap_score_array)

    device = target_span_marginals_array.device
    num_pas_samples = [cnt[1] for cnt in sorted(Counter(pasid2groupid_map).items())]
    assert len(target_spanoverlap_score_array) == sum(
        num_pas_samples
    ), "In the current implementation, we only allow num_pas_samples = 1"

    assert not torch.any(torch.isnan(target_span_marginals_array)) and not torch.any(
        torch.isnan(pas_span_marginals_array)
    )

    def inner_product_score(trg_marginals_pack, hit_count_dict):
        # assuming in log space already

        trg_marginals, trg_indices = trg_marginals_pack
        device = trg_marginals.device

        alignment_array = torch.zeros(len(trg_indices[0]), device=device)

        for alignment_idx, span in enumerate(torch.stack(trg_indices, dim=1)):
            span = tuple(span.tolist())
            if span in hit_count_dict.keys():
                alignment_array[alignment_idx] = hit_count_dict[span]

        tmp1 = torch.where(alignment_array - 2 > 0, alignment_array - 2, 0.0)
        activation = tmp1 / (tmp1 + 1)  # will be from 0 to 1
        alignment_array = torch.bernoulli(activation)

        num_hits = (alignment_array > 0).sum().float()

        score = -torch.sum(trg_marginals * alignment_array).squeeze()
        score = (score / num_hits) if num_hits > 0 else torch.tensor(0.0, device=device)
        # print(score)
        # input()

        return score, num_hits

    target_span_marginals_array = [
        get_normalized_span_prob(
            trg_marginals.sum(-1),
            flag_log_mode=True,
            flag_normalize_prob=flag_normalize_marginals,
            max_bandwidth=max_bandwidth,
        )
        for trg_marginals in target_span_marginals_array
    ]
    padded_target_span_marginals_array = [
        [trg_marginals for i in range(num_ps)]
        for trg_marginals, num_ps in zip(target_span_marginals_array, num_pas_samples)
    ]
    padded_target_span_marginals_array = list(
        itertools.chain.from_iterable(padded_target_span_marginals_array)
    )
    pas_span_marginals_array = [
        get_normalized_span_prob(
            pas_marginals.sum(-1),
            flag_log_mode=True,
            flag_normalize_prob=flag_normalize_marginals,
            max_bandwidth=max_bandwidth,
        )
        for pas_marginals in pas_span_marginals_array
    ]

    l_alignment_outputs = [
        inner_product_score(trg_marginals, hit_count_dict)
        for trg_marginals, hit_count_dict in zip(
            padded_target_span_marginals_array, target_spanoverlap_score_array
        )
    ]
    lscore, lnum_hits = zip(*l_alignment_outputs)
    r_alignment_outputs = [
        inner_product_score(pas_marginals, hit_count_dict)
        for pas_marginals, hit_count_dict in zip(
            pas_span_marginals_array, pas_spanoverlap_score_array
        )
    ]
    rscore, rnum_hits = zip(*r_alignment_outputs)

    lscore_array = [
        torch.mean(arr) for arr in torch.stack(lscore, dim=0).split(num_pas_samples)
    ]
    rscore_array = [
        torch.mean(arr) for arr in torch.stack(rscore, dim=0).split(num_pas_samples)
    ]
    lnum_hits_array = [
        torch.mean(arr) for arr in torch.stack(lnum_hits, dim=0).split(num_pas_samples)
    ]
    rnum_hits_array = [
        torch.mean(arr) for arr in torch.stack(rnum_hits, dim=0).split(num_pas_samples)
    ]

    avg_lscore_per_group = torch.stack(lscore_array, dim=0)
    avg_rscore_per_group = torch.stack(rscore_array, dim=0)
    avg_lnum_hits_per_group = torch.stack(lnum_hits_array, dim=0)
    avg_rnum_hits_per_group = torch.stack(rnum_hits_array, dim=0)

    return (
        avg_lscore_per_group,
        avg_rscore_per_group,
        avg_lnum_hits_per_group,
        avg_rnum_hits_per_group,
    )


def compute_oneside_matching_with_spanoverlap_score_competing(
    merged_alignment_array,
    merged_span_marginals_array_raw,
    competing_span_mask,
    flag_normalize_marginals,
    spanoverlap_mask_array=None,
    max_bandwidth=1000,
    hit_count_threshold=2,
    activation_flood=4,
    use_SN=False,
):
    from ipynb.spanoverlap import SpanScorer

    scorer = SpanScorer()
    from collections import Counter
    import itertools

    device = merged_span_marginals_array_raw.device

    merged_span_marginals_array = [
        get_normalized_span_prob(
            trg_marginals.sum(-1),
            flag_log_mode=True,
            flag_normalize_prob=flag_normalize_marginals,
            max_bandwidth=max_bandwidth,
        )[0]
        for trg_marginals in merged_span_marginals_array_raw
    ]
    merged_span_marginals_array = torch.stack(merged_span_marginals_array, dim=0)

    pairwise_alignment_array = merged_alignment_array.unsqueeze(
        2
    ) - merged_alignment_array.unsqueeze(1)
    # print(pairwise_alignment_array.shape, competing_span_mask.shape)
    pairwise_alignment_activation = (
        torch.where(pairwise_alignment_array > hit_count_threshold, 1, 0.0)
        * competing_span_mask.float()
    )
    pairwise_posterior_gap = torch.maximum(
        torch.zeros_like(pairwise_alignment_activation, device=device),
        -(
            merged_span_marginals_array.unsqueeze(2)
            - merged_span_marginals_array.unsqueeze(1)
        )
        + 1,
    )
    num_hits = ((pairwise_alignment_activation != 0).sum(-1) > 0).sum(-1)
    scores = (pairwise_alignment_activation * pairwise_posterior_gap).topk(
        k=min(5, pairwise_posterior_gap.shape[-1]), dim=-1
    ).values.mean(-1).sum(-1) / (num_hits + 1e-8)

    scores = pairwise_alignment_activation * pairwise_posterior_gap
    topk_scores = scores.topk(k=min(5, scores.shape[-1]), dim=-1).values
    topk_nonzero_cnt = (topk_scores > 0).sum(-1)
    scores = (topk_scores.sum(-1) / (topk_nonzero_cnt + 1e-8)).sum(-1) / (
        num_hits + 1e-8
    )

    # print(scores)
    # input()

    # scores = scores / (num_hits+1e-8)

    return scores, torch.clamp(num_hits, min=0, max=1)


def compute_oneside_matching_with_spanoverlap_score_competing_v2(
    merged_alignment_array,
    merged_span_marginals_array_raw,
    competing_span_mask,
    flag_normalize_marginals,
    spanoverlap_mask_array=None,
    max_bandwidth=1000,
    hit_count_threshold=2,
    activation_flood=4,
    use_SN=False,
):
    from ipynb.spanoverlap import SpanScorer

    scorer = SpanScorer()
    from collections import Counter
    import itertools

    device = merged_span_marginals_array_raw.device

    merged_span_marginals_array = [
        get_normalized_span_prob(
            trg_marginals.sum(-1),
            flag_log_mode=True,
            flag_normalize_prob=flag_normalize_marginals,
            max_bandwidth=max_bandwidth,
        )[0]
        for trg_marginals in merged_span_marginals_array_raw
    ]
    merged_span_marginals_array = torch.stack(merged_span_marginals_array, dim=0)

    pairwise_alignment_array = torch.clamp(
        merged_alignment_array.unsqueeze(2) - merged_alignment_array.unsqueeze(1), min=0
    )
    # print(pairwise_alignment_array.shape, competing_span_mask.shape)
    pairwise_alignment_activation = (
        torch.bernoulli(
            pairwise_alignment_array / (pairwise_alignment_array + hit_count_threshold)
        )
        * competing_span_mask.float()
    )
    pairwise_posterior_gap = torch.maximum(
        torch.zeros_like(pairwise_alignment_activation, device=device),
        -(
            merged_span_marginals_array.unsqueeze(2)
            - merged_span_marginals_array.unsqueeze(1)
        )
        + 1,
    )
    # print(pairwise_posterior_gap)
    num_hits = ((pairwise_alignment_activation != 0).sum(-1) > 0).sum(-1)
    scores = (pairwise_alignment_activation * pairwise_posterior_gap).topk(
        k=min(5, pairwise_posterior_gap.shape[-1]), dim=-1
    ).values.mean(-1).sum(-1) / (num_hits + 1e-8)

    scores = pairwise_alignment_activation * pairwise_posterior_gap
    topk_scores = scores.topk(k=min(5, scores.shape[-1]), dim=-1).values
    topk_nonzero_cnt = (topk_scores > 0).sum(-1)
    scores = (topk_scores.sum(-1) / (topk_nonzero_cnt + 1e-8)).sum(-1) / (
        num_hits + 1e-8
    )

    # print(scores)
    # input()

    # scores = scores / (num_hits+1e-8)

    return scores, torch.clamp(num_hits, min=0, max=1)


def compute_oneside_matching_with_spanoverlap_score_competing_freq(
    merged_alignment_array,
    merged_span_marginals_array_raw,
    competing_span_mask,
    flag_normalize_marginals,
    spanoverlap_mask_array=None,
    max_bandwidth=1000,
    hit_count_threshold=2,
    activation_flood=4,
    use_SN=False,
):
    from ipynb.spanoverlap import SpanScorer

    scorer = SpanScorer()
    from collections import Counter
    import itertools

    device = merged_span_marginals_array_raw.device
    merged_wo_info_array = -(merged_alignment_array+1e-9).log()#-torch.clamp(merged_alignment_array.log(), min=-1e9)

    info_admissible_array = (merged_wo_info_array < 3).float()
    info_admissible_array = info_admissible_array.unsqueeze(
        2
    ) * info_admissible_array.unsqueeze(
        1
    )  # 1: admissible, 0: inadmissible
    info_admissible_array = (
        info_admissible_array * competing_span_mask.float()
    )  # needs to both be admissible and competing

    merged_span_marginals_array = [
        get_normalized_span_prob(
            trg_marginals.sum(-1),
            flag_log_mode=True,
            flag_normalize_prob=flag_normalize_marginals,
            max_bandwidth=max_bandwidth,
        )[0]
        for trg_marginals in merged_span_marginals_array_raw
    ]
    merged_span_marginals_array = torch.stack(merged_span_marginals_array, dim=0)
    pairwise_posterior_gap = torch.maximum(
        torch.zeros_like(info_admissible_array, device=device),
        -(
            merged_span_marginals_array.unsqueeze(2)
            - merged_span_marginals_array.unsqueeze(1)
        )
        + 1,
    )

    info_gain_array = torch.where(
        -merged_wo_info_array.unsqueeze(2) + merged_wo_info_array.unsqueeze(1)> 1,
        1., 0.
    ) #establish a margin here  # info gain array
    # print(
    #     info_gain_array.shape, info_admissible_array.shape, info_admissible_array.shape
    # )
    scores = ((info_gain_array * pairwise_posterior_gap) * info_admissible_array).sum(
        -1
    ) / (
        ((info_gain_array * info_admissible_array).sum(-1) + 1e-8).float()
    )  # (b, n)
    num_hits = ((info_admissible_array*info_gain_array).sum(-1) > 0).sum(-1).float()  # (b,)
    scores = scores.sum(-1) / (num_hits + 1e-8)
    # print("info_gain_array", info_gain_array)

    # print(pairwise_alignment_array.shape, competing_span_mask.shape)
    # # pairwise_alignment_activation = torch.bernoulli(pairwise_alignment_array/(pairwise_alignment_array+hit_count_threshold)) * competing_span_mask.float()
    # span_info_gain = (
    #     merged_span_marginals_array * info_gain_array
    # )  # p((i, j)) * max(0, log(f(i, j)) - log(f(m,n))-1)
    # num_hits = (info_gain_array.sum(-1) > 0).sum(-1)
    # scores = span_info_gain.sum(-1) / (num_hits + 1e-8)

    return scores, torch.clamp(num_hits, min=0, max=1)


def compute_oneside_matching_with_spanoverlap_score_merged_v2(
    merged_alignment_array,
    merged_span_marginals_array,
    flag_normalize_marginals,
    spanoverlap_mask_array=None,
    max_bandwidth=1000,
    hit_count_threshold=2,
    activation_flood=4,
    use_SN=False,
    use_hard_score=False,
):
    from ipynb.spanoverlap import SpanScorer
    from collections import Counter
    import itertools

    # print(merged_span_marginals_array.shape)
    assert not torch.any(torch.isnan(merged_span_marginals_array))  # and not torch.any(

    # if not use_SN:
    merged_span_marginals_array = torch.stack(
        [
            get_normalized_span_prob(
                trg_marginals.sum(-1),
                flag_log_mode=True,
                flag_normalize_prob=flag_normalize_marginals,
                max_bandwidth=max_bandwidth,
            )[0]
            for trg_marginals in merged_span_marginals_array
        ],
        dim=0,
    )

    alignment_array = torch.bernoulli(
        merged_alignment_array / (merged_alignment_array + hit_count_threshold)
    )

    # print(merged_span_marginals_array.shape, alignment_array.shape)
    merged_span_marginals_array = torch.minimum(
        merged_span_marginals_array - 10,
        merged_span_marginals_array.new_zeros(*merged_span_marginals_array.shape),
    )
    scores = -torch.sum(merged_span_marginals_array * alignment_array, dim=-1)
    num_hits = (alignment_array > 0).sum(-1).float()
    scores = scores / (num_hits + 1e-8)

    return scores, torch.clamp(num_hits, min=0, max=1)


def compute_oneside_matching_with_spanoverlap_score_merged_freq(
    merged_alignment_array,
    merged_span_marginals_array,
    flag_normalize_marginals,
    spanoverlap_mask_array=None,
    max_bandwidth=1000,
    hit_count_threshold=2,
    activation_flood=4,
    use_SN=False,
    use_hard_score=False,
):
    from ipynb.spanoverlap import SpanScorer
    from collections import Counter
    import itertools

    # print(merged_span_marginals_array.shape)
    assert not torch.any(torch.isnan(merged_span_marginals_array))  # and not torch.any(
    # print(merged_alignment_array)
    # wo_info_array = -torch.clamp(merged_alignment_array.log(), min=-1e9)
    # assert (merged_alignment_array>=0).all() and (merged_alignment_array<=1).all(), f"merged_alignment_array must be a frequency array, but it is {merged_alignment_array}"
    freq_array = torch.bernoulli(torch.clamp(merged_alignment_array, min=0, max=1))

    # if not use_SN:
    merged_span_marginals_array = torch.stack(
        [
            get_normalized_span_prob(
                trg_marginals.sum(-1),
                flag_log_mode=True,
                flag_normalize_prob=flag_normalize_marginals,
                max_bandwidth=max_bandwidth,
            )[0]
            for trg_marginals in merged_span_marginals_array
        ],
        dim=0,
    )
    # print("wo info array", wo_info_array)

    # info_admissible_array = (wo_info_array < 8).float()
    info_cost = (
        -merged_span_marginals_array * freq_array# * info_admissible_array
    )
    scores = info_cost.sum(-1)

    # alignment_array = torch.maximum(
    #     torch.clamp(merged_alignment_array.log(), min=-1e9) + 8,
    #     merged_alignment_array.new_zeros(*merged_alignment_array.shape),
    # )
    # alignment_array = torch.bernoulli(merged_alignment_array/(merged_alignment_array+hit_count_threshold))

    # print(merged_span_marginals_array.shape, alignment_array.shape)
    # merged_span_marginals_array = torch.minimum(merged_span_marginals_array-10, merged_span_marginals_array.new_zeros(*merged_span_marginals_array.shape))
    # scores = torch.sum(merged_span_marginals_array * alignment_array, dim=-1)
    # scores = -torch.sum(merged_span_marginals_array * alignment_array, dim=-1)
    num_hits = freq_array.sum(-1).float()
    scores = scores / (num_hits + 1e-8)

    return scores, torch.clamp(num_hits, min=0, max=1)


def compute_oneside_matching_with_spanoverlap_score_merged(
    merged_alignment_array,
    merged_span_marginals_array,
    flag_normalize_marginals,
    spanoverlap_mask_array=None,
    max_bandwidth=1000,
    hit_count_threshold=2,
    activation_flood=4,
    use_SN=False,
    use_hard_score=False,
):
    from ipynb.spanoverlap import SpanScorer
    from collections import Counter
    import itertools

    # print(merged_span_marginals_array.shape)
    assert not torch.any(torch.isnan(merged_span_marginals_array))  # and not torch.any(

    # if not use_SN:
    merged_span_marginals_array = torch.stack(
        [
            get_normalized_span_prob(
                trg_marginals.sum(-1),
                flag_log_mode=True,
                flag_normalize_prob=flag_normalize_marginals,
                max_bandwidth=max_bandwidth,
            )[0]
            for trg_marginals in merged_span_marginals_array
        ],
        dim=0,
    )
    # else:
    # merged_span_marginals_array = torch.stack([get_normalized_span_prob_SN(trg_marginals.sum(-1), flag_log_mode=True, flag_normalize_prob=flag_normalize_marginals, max_bandwidth=max_bandwidth)[0] for trg_marginals in merged_span_marginals_array], dim=0)

    if use_hard_score:
        alignment_array = torch.clamp(
            torch.maximum(
                merged_alignment_array - hit_count_threshold,
                merged_alignment_array.new_zeros(*merged_alignment_array.shape),
            ),
            min=0.0,
            max=1.0,
        )
    else:
        # tmp = torch.where(merged_alignment_array-hit_count_threshold>0, merged_alignment_array-hit_count_threshold, 0.)
        # alignment_array = torch.bernoulli(tmp/(tmp+activation_flood)) #will be from 0 to 1
        alignment_array = torch.clamp(
            merged_alignment_array - hit_count_threshold, max=1.0, min=0.0
        )

    # print(merged_span_marginals_array.shape, alignment_array.shape)
    merged_span_marginals_array = torch.minimum(
        merged_span_marginals_array - 10,
        merged_span_marginals_array.new_zeros(*merged_span_marginals_array.shape),
    )
    scores = -torch.sum(merged_span_marginals_array * alignment_array, dim=-1)
    num_hits = (alignment_array > 0).sum(-1).float()
    scores = scores / (num_hits + 1e-8)

    return scores, torch.clamp(num_hits, min=0, max=1)


def compute_inverse_matching_distance(
    target_form_array,
    pas_form_array,
    pas_len_array,
    target_span_marginals_array,
    pas_span_marginals_array,
    pasid2groupid_map,
    flag_normalize_marginals,
    max_bandwidth=4,
    return_negative_score=False,
):
    from ipynb.spanoverlap import SpanScorer

    scorer = SpanScorer()
    from collections import Counter
    import itertools

    batch_size = len(target_form_array)

    device = target_span_marginals_array.device
    num_pas_samples = [cnt[1] for cnt in sorted(Counter(pasid2groupid_map).items())]

    if torch.any(torch.isnan(target_span_marginals_array)) or torch.any(
        torch.isnan(pas_span_marginals_array)
    ):
        print("nan in span marginals")
        print(target_form_array, pas_form_array)
        # print(target_span_marginals_array)
        # print(pas_span_marginals_array)
    assert not torch.any(torch.isnan(target_span_marginals_array)) and not torch.any(
        torch.isnan(pas_span_marginals_array)
    )

    padded_target_form_array = [
        [trg_form for i in range(num_ps)]
        for trg_form, num_ps in zip(target_form_array, num_pas_samples)
    ]
    padded_target_form_array = list(
        itertools.chain.from_iterable(padded_target_form_array)
    )
    alignment_mtx_array = [
        scorer.score_by_longest_matches(
            trg_form, sample_form, stemmer, flag_print=False
        )
        for trg_form, sample_form in zip(padded_target_form_array, pas_form_array)
    ]

    # scorer.score_by_count(form, sample, stemmer)

    # def get_normalized_span_prob(span_marginals):
    #     # print(span_marginals.shape)
    #     # print(span_marginals)
    #     masklen = len(span_marginals)
    #     mask_1 = torch.arange(masklen, device = device).float()[None, :].expand(masklen, masklen)
    #     mask_2 = mask_1.T
    #     mask = (mask_2 - mask_1).abs()<=max_bandwidth
    #     indices = (span_marginals * mask).nonzero(as_tuple=True)
    #     nonzero_marginals = span_marginals[
    #         indices
    #     ]
    #     # print(nonzero_marginals.shape)
    #     return nonzero_marginals / nonzero_marginals.sum(), indices

    def inner_product_score(trg_marginals_pack, alignment_mtx, pas_marginals_pack):

        trg_marginals, trg_indices = trg_marginals_pack
        pas_marginals, pas_indices = pas_marginals_pack
        alignment_mtx = (
            torch.from_numpy(alignment_mtx)
            .to(device)[trg_indices]
            .permute(1, 2, 0)[pas_indices]
            .permute(1, 0)
            .float()
        )

        score = -torch.sum(
            (
                (trg_marginals.unsqueeze(1) + pas_marginals.unsqueeze(0))
                * alignment_mtx
            ).flatten(),
            dim=0,
        )
        score = score.squeeze()
        return score

    input_pas_span_marginals_array = pas_span_marginals_array
    target_span_marginals_array = [
        get_normalized_span_prob(
            trg_marginals.sum(-1),
            max_bandwidth=max_bandwidth,
            flag_normalize_prob=flag_normalize_marginals,
            flag_log_mode=True,
        )
        for trg_marginals in target_span_marginals_array
    ]
    padded_target_span_marginals_array = [
        [trg_marginals for i in range(num_ps)]
        for trg_marginals, num_ps in zip(target_span_marginals_array, num_pas_samples)
    ]
    padded_target_span_marginals_array = list(
        itertools.chain.from_iterable(padded_target_span_marginals_array)
    )
    pas_span_marginals_array = [
        get_normalized_span_prob(
            pas_marginals.sum(-1),
            max_bandwidth=max_bandwidth,
            flag_normalize_prob=flag_normalize_marginals,
            flag_log_mode=True,
        )
        for pas_marginals in pas_span_marginals_array
    ]

    alignment_score = torch.stack(
        [
            inner_product_score(trg_marginals, alignment_mtx, pas_marginals)
            for trg_marginals, alignment_mtx, pas_marginals in zip(
                padded_target_span_marginals_array,
                alignment_mtx_array,
                pas_span_marginals_array,
            )
        ],
        dim=0,
    )
    group_alignment_score = alignment_score.split(num_pas_samples)
    score_array = [torch.mean(arr) for arr in group_alignment_score]
    avg_score_per_group = torch.stack(score_array, dim=0)

    if return_negative_score:
        original_index_tag = [
            i for i, num_ps in enumerate(num_pas_samples) for j in range(num_ps)
        ]
        zipped_sm_sf = list(
            zip(
                list(range(len(input_pas_span_marginals_array))),
                pas_form_array,
                original_index_tag,
            )
        )
        # random.shuffle(zipped_sm_sf)
        shuffled_sm_sf = []
        for orig_tag in original_index_tag:
            smsf_option_list = [smsf for smsf in zipped_sm_sf if smsf[2] != orig_tag]
            if len(smsf_option_list) == 0:
                pas_span_marginal_idx, pas_form, orig_tag = random.choice(
                    [smsf for smsf in zipped_sm_sf]
                )
            else:
                pas_span_marginal_idx, pas_form, orig_tag = random.choice(
                    smsf_option_list
                )
            smsf_options = (
                input_pas_span_marginals_array[pas_span_marginal_idx],
                pas_form,
                orig_tag,
            )
            shuffled_sm_sf.append(smsf_options)
            zipped_sm_sf.remove((pas_span_marginal_idx, pas_form, orig_tag))
        zipped_sm_sf = shuffled_sm_sf

        neg_pas_span_marginals_array, neg_pas_form_array, _ = zip(*zipped_sm_sf)
        neg_pas_span_marginals_array = [
            get_normalized_span_prob(
                pas_marginals,
                max_bandwidth=max_bandwidth,
                flag_normalize_prob=flag_normalize_marginals,
                flag_log_mode=True,
            )
            for pas_marginals in neg_pas_span_marginals_array
        ]
        # neg_alignment_score = torch.stack([inner_product_score(trg_marginals, alignment_mtx, pas_marginals) for trg_marginals, alignment_mtx, pas_marginals in zip(padded_target_span_marginals_array, alignment_mtx_array, neg_pas_span_marginals_array)], dim=0)
        neg_alignment_mtx_array = [
            scorer.score_by_longest_matches(trg_form, sample_form, stemmer)
            for trg_form, sample_form in zip(
                padded_target_form_array, neg_pas_form_array
            )
        ]
        neg_alignment_score = torch.stack(
            [
                inner_product_score(trg_marginals, alignment_mtx, pas_marginals)
                for trg_marginals, alignment_mtx, pas_marginals in zip(
                    padded_target_span_marginals_array,
                    neg_alignment_mtx_array,
                    neg_pas_span_marginals_array,
                )
            ],
            dim=0,
        )

        # neg_pas_
        neg_group_alignment_score = neg_alignment_score.split(num_pas_samples)
        neg_score_array = [torch.mean(arr) for arr in neg_group_alignment_score]
        avg_score_per_neg_group = torch.stack(neg_score_array, dim=0)
        return avg_score_per_group, avg_score_per_neg_group

    # print(avg_score_per_group)
    return avg_score_per_group, torch.tensor([0.0])


def convert_ptb_words_and_spans_to_spacy_tokens_and_spans(words, spans, nlp):

    def align_tok_with_word(tokenized, word):
        token_groups = [[]]
        current_word_idx = 0

        for i, token in enumerate(tokenized):
            tok, pos, whitespace, lemma = token
            if whitespace == " ":
                token_groups[-1].append(i)
                token_groups.append([])
            else:
                token_groups[-1].append(i)
        if len(token_groups[-1]) == 0:
            token_groups = token_groups[:-1]
        token_groups += [[token_groups[-1][-1] + 1]]
        return token_groups

    def replace_parentheses(words):
        for i, w in enumerate(words):
            if w == "-lrb-":
                words[i] = "("
            if w == "-rrb-":
                words[i] = ")"
        return words

    def convert_spans_to_spacy(spans, w2t_mapping, tokens):
        tokens = [t[0] for t in tokens]
        new_spans = []
        for s in spans:
            start, end, label = s
            # print(s)
            # print(w2t_mapping[start], w2t_mapping[end])
            new_spans.append((w2t_mapping[start][0], w2t_mapping[end][0], label))

        # for s in new_spans:

        # print(' '.join(tokens[s[0]:s[1]]), s)
        return new_spans

    words = replace_parentheses(words)

    doc = nlp(" ".join(words))
    spacy_tokens = [(w.text, w.pos_, w.whitespace_, w.lemma_.lower()) for w in doc]
    w2t_mapping = align_tok_with_word(spacy_tokens, words)
    spacy_spans = convert_spans_to_spacy(spans, w2t_mapping, spacy_tokens)
    return spacy_tokens, spacy_spans, w2t_mapping


def convert_spacy_span_to_fit_ptb_tokenization(spacy_spans, w2t_mapping):
    normalized_spans = []
    # print(spacy_spans)
    for s in spacy_spans:
        l, r = s[:2]
        for gid in range(len(w2t_mapping)):
            g = w2t_mapping[gid]
            if g[0] <= l < g[-1] + 1:
                l = gid
            if g[0] <= r < g[-1] + 1:
                r = gid
            # if g[0] <= l < g[-1]+1:
            #     l = g[0]
            # if g[0] == r < g[-1]+1:
            #     r = g[0]
            # if g[0] < r < g[-1]+1:
            #     r = w2t_mapping[gid+1][0]
        normalized_spans.append((l, r))
    return normalized_spans

def so_accumulation_tbtok(seq_array, stemmer, flag_compute_relative_frequency=False, corpus=None, langstr='none'):
    # return an array of counts where counts are the number of occurrence of this substring in the sequences.
    count_array = []
    seq_array = [[stemmer.stemWord(removing_punctuations(w).strip()).lower() for w in seq] for seq in seq_array]
    assert langstr != 'none', "Please specify the language of the corpus"
    # print(seq_array)
    sentence_joiner = " " if langstr != 'chinese' else ""

    for refidx, ref in enumerate(seq_array):
        cmp_set = [sentence_joiner.join([w for w in seq if len(w)>0]) for seq in seq_array if seq != ref]
        cmp_word_set = [set(seq) for seq in seq_array if seq != ref]
        substring_count = {}
        if refidx>0:
            count_array.append(substring_count)
            continue

        # print(ref, cmp_set[:2])

        hitmap = [[] for _ in range(len(cmp_set))]
        for w in range(len(ref) - 1, 1, -1):
            for i in range(len(ref) - w + 1):
                substr = [word for word in ref[i : i + w] if len(word)>0]
                substring = sentence_joiner.join(substr)
                substring_word_set = set(substr)
                # print(substring)
                count_full = 0
                ws_count_full = 0
                count_limited = 0
                ws_count_limited = 0
                for cmp_idx, (cmp, cmp_ws) in enumerate(zip(cmp_set, cmp_word_set)):
                    hit_check = [
                        hit[0] <= i and hit[1] >= i + w for hit in hitmap[cmp_idx]
                    ]
                    if substring_word_set <= cmp_ws:
                        ws_count_full += 1
                    if substring in cmp:
                        count_full += 1
                        # hitmap[cmp_idx].append((i, i + w))
                    if any(hit_check):
                        continue
                    # print(substring, cmp)
                    if substring_word_set <= cmp_ws:
                        ws_count_limited += 1
                    if substring in cmp:
                        count_limited += 1
                        hitmap[cmp_idx].append((i, i + w))
                corpus_size = len(corpus)
                corpus_freq = sum([1 for doc in corpus if substring in doc])
                # print(corpus_freq, corpus_size)

                if count_limited > 0 or count_full > 0:
                    if flag_compute_relative_frequency:
                        substring_count[(i, i + w)] = (count_full, count_limited, ws_count_full, ws_count_limited, corpus_size, corpus_freq)
                    else:
                        substring_count[(i, i + w)] = count_limited
        substring_count[(0, len(ref))] = 1
        count_array.append(substring_count)
        # break

    return count_array



def so_accumulation(seq_array, stemmer, flag_compute_relative_frequency=False):
    # return an array of counts where counts are the number of occurrence of this substring in the sequences.
    count_array = []
    seq_array = [[stemmer.stemWord(w).lower() for w in seq] for seq in seq_array]
    # print(seq_array)

    for ref in seq_array:
        cmp_set = ["[SEP]"+"[SEP]".join(seq)+"[SEP]" for seq in seq_array if seq != ref]
        cmp_word_set = [set(seq) for seq in seq_array if seq != ref]
        substring_count = {}

        hitmap = [[] for _ in range(len(cmp_set))]
        for w in range(len(ref) - 1, 1, -1):
            for i in range(len(ref) - w + 1):
                substring = "[SEP]"+"[SEP]".join(ref[i : i + w])+"[SEP]"
                substring_word_set = set(ref[i : i + w])
                # print(substring)
                count_full = 0
                ws_count_full = 0
                count_limited = 0
                ws_count_limited = 0
                for cmp_idx, (cmp, cmp_ws) in enumerate(zip(cmp_set, cmp_word_set)):
                    hit_check = [
                        hit[0] <= i and hit[1] >= i + w for hit in hitmap[cmp_idx]
                    ]
                    if substring_word_set <= cmp_ws:
                        ws_count_full += 1
                    if substring in cmp:
                        count_full += 1
                        # hitmap[cmp_idx].append((i, i + w))
                    if any(hit_check):
                        continue
                    # print(substring, cmp)
                    if substring_word_set <= cmp_ws:
                        ws_count_limited += 1
                    if substring in cmp:
                        count_limited += 1
                        hitmap[cmp_idx].append((i, i + w))
                if count_limited > 0 or count_full > 0:
                    if flag_compute_relative_frequency:
                        substring_count[(i, i + w)] = (count_full, count_limited, ws_count_full, ws_count_limited)
                    else:
                        substring_count[(i, i + w)] = count_limited
        substring_count[(0, len(ref))] = 1
        count_array.append(substring_count)

    return count_array


import math


class CosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.00001,
        eta_min: float = 0.00001,
        last_epoch: int = -1,
    ):
        """
        Args:
            optimizer (torch.optim.Optimizer):
                最適化手法インスタンス
            warmup_epochs (int):
                linear warmupを行うepoch数
            max_epochs (int):
                cosine曲線の終了に用いる 学習のepoch数
            warmup_start_lr (float):
                linear warmup 0 epoch目の学習率
            eta_min (float):
                cosine曲線の下限
            last_epoch (int):
                cosine曲線の位相オフセット
        学習率をmax_epochsに至るまでコサイン曲線に沿ってスケジュールする
        epoch 0からwarmup_epochsまでの学習曲線は線形warmupがかかる
        https://pytorch-lightning-bolts.readthedocs.io/en/stable/schedulers/warmup_cosine_annealing.html
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
        return None

    def get_lr(self):
        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        if self.last_epoch < self.warmup_epochs:
            return [
                group["lr"]
                + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        if self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        if (self.last_epoch - 1 - self.max_epochs) % (
            2 * (self.max_epochs - self.warmup_epochs)
        ) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min)
                * (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs)))
                / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            / (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs - 1)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]
