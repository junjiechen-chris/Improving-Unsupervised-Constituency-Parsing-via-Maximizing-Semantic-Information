# -*- coding: utf-8 -*-

from collections import Counter, defaultdict
import torch


class Metric(object):

    def __lt__(self, other):
        return self.score < other

    def __le__(self, other):
        return self.score <= other

    def __ge__(self, other):
        return self.score >= other

    def __gt__(self, other):
        return self.score > other

    @property
    def score(self):
        return -1e9


import numpy as np


def uf1_tpfpfn(preds, golds):
    tpfpfn_array = []
    for pred, gold in zip(preds, golds):
        if len(pred) == 0:
            tpfpfn_array.append(np.array([0, 0, 0]))
        length = max(gold, key=lambda x: x[1])[1]
        # removing the trival span
        gold = list(filter(lambda x: x[0] + 1 != x[1], gold))
        pred = list(filter(lambda x: x[0] + 1 != x[1], pred))
        # remove the entire sentence span.
        gold = list(filter(lambda x: not (x[0] == 0 and x[1] == length), gold))
        pred = list(filter(lambda x: not (x[0] == 0 and x[1] == length), pred))
        # remove label.
        gold = [g[:2] for g in gold]
        pred = [p[:2] for p in pred]
        gold = list(map(tuple, gold))
        # corpus f1
        gold = set(gold)
        pred = set(pred)
        overlap = pred.intersection(gold)
        tp, fp, fn = len(overlap), len(pred) - len(overlap), len(gold) - len(overlap)

        tpfpfn_array.append(np.array([tp, fp, fn]))
    return np.stack(tpfpfn_array, axis=0)


def test_if_span_overlap(span1, span2):
    # This function is used to test if two spans overlap but does not contain each other.
    # span1 and span2 are both tuples.
    c1 = span1[0] < span2[0] and span1[1] > span2[0] and span1[1] < span2[1]
    c2 = span1[0] > span2[0] and span1[0] < span2[1] and span1[1] > span2[1]
    return c1 or c2


class RewardAccumulator(Metric):
    def __init__(self, eps=1e-8) -> None:
        super(RewardAccumulator, self).__init__()
        self.reward_list = []
        self.eps = eps
        self.n = 0

    def __call__(self, trees, reward):
        reward = reward.cpu().detach()
        reward = torch.maximum(reward, reward.new_zeros(reward.shape))
        for t, r in zip(trees, reward):
            racc = 0
            for s in t:
                racc += r[s[0], s[1]]
            self.reward_list.append(racc)
            self.n += 1

    @property
    def average_reward(self):
        return sum(self.reward_list) / self.n if self.n > 0 else 0


class UF1(Metric):
    def __init__(self, eps=1e-8, device=torch.device("cuda"), log_file=None):
        super(UF1, self).__init__()
        self.f1 = 0.0
        self.evalb = 0.0
        self.n = 0.0
        self.eps = eps
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.device = device
        self.log_file = log_file

    def __call__(self, preds, golds, log_metric=None):
        f1_list = []
        # print(self.log_file, log_metric)
        assert (log_metric is None) == (
            self.log_file is None
        ), "log_metric and log_file should be both None or both not None."
        if log_metric is None:
            log_metric = ["null"] * len(preds)
        assert len(preds) == len(
            log_metric
        ), "preds and log_metric should have same length."
        for pred, gold, metric in zip(preds, golds, log_metric):
            # in the case of sentence length=1
            if len(pred) == 0:
                continue
            # print(gold)
            length = max(gold, key=lambda x: x[1])[1]
            
            # removing the trival span
            gold = list(filter(lambda x: x[0] + 1 != x[1] and x[0] != x[1], gold))
            pred = list(filter(lambda x: x[0] + 1 != x[1] and x[0] != x[1], pred))
            # remove the entire sentence span.
            gold = list(filter(lambda x: not (x[0] == 0 and x[1] == length), gold))
            pred = list(filter(lambda x: not (x[0] == 0 and x[1] == length), pred))
            # remove label.
            gold = [g[:2] for g in gold]
            pred = [p[:2] for p in pred]
            gold = list(map(tuple, gold))
            # corpus f1
            for span in pred:
                if span in gold:
                    self.tp += 1
                else:
                    self.fp += 1
            for span in gold:
                if span not in pred:
                    self.fn += 1

            # sentence f1
            # remove duplicated span.
            gold = set(gold)
            pred = set(pred)
            # print(len(gold), len(pred))
            overlap = pred.intersection(gold)
            prec = float(len(overlap)) / (len(pred) + self.eps)
            reca = float(len(overlap)) / (len(gold) + self.eps)
            if len(gold) == 0:
                reca = 1.0
                if len(pred) == 0:
                    prec = 1.0
            f1 = 2 * prec * reca / (prec + reca + 1e-8)
            f1_list.append(f1)
            if self.log_file is not None:
                # print(metric)
                with open(self.log_file, "a") as f:
                    out_str = "\t".join(
                        [
                            (
                                str(m.cpu().item())
                                if isinstance(m, torch.Tensor)
                                else str(m)
                            )
                            for m in metric
                        ]
                    )
                    f.write(f"{f1}\t{out_str}\n")
            self.f1 += f1
            self.n += 1
        return f1_list

    @property
    def sentence_uf1(self):
        print("total samples", self.n)
        return self.f1 / self.n

    @property
    def corpus_uf1(self):
        if self.tp == 0 and self.fp == 0:
            return 0

        prec = self.tp / (self.tp + self.fp)
        recall = self.tp / (self.tp + self.fn)
        corpus_f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
        return corpus_f1

    @property
    def corpus_precision(self):
        if self.tp == 0 and self.fp == 0:
            return 0
        return self.tp / (self.tp + self.fp)

    @property
    def corpus_recall(self):
        if self.tp == 0 and self.fn == 0:
            return 0
        return self.tp / (self.tp + self.fn)

    @property
    def score(self):
        return self.sentence_uf1

    def __repr__(self):
        s = f"Sentence F1: {self.sentence_uf1:6.2%} Corpus F1: {self.corpus_uf1:6.2%} Corpus Precision: {self.corpus_precision:6.2%} Corpus Recall: {self.corpus_recall:6.2%}"
        return s


class UAS(Metric):
    def __init__(self, eps=1e-8):
        super(Metric, self).__init__()
        self.eps = eps
        self.total = 0.0
        self.direct_correct = 0.0
        self.undirect_correct = 0.0
        self.total_sentence = 0.0
        self.correct_root = 0.0

    @property
    def score(self):
        return self.direct_correct / self.total

    def __call__(self, predicted_arcs, gold_arcs):

        for pred, gold in zip(predicted_arcs, gold_arcs):
            assert len(pred) == len(gold)

            if len(pred) > 0:
                self.total_sentence += 1.0

            for head, child in pred:
                if gold[int(child)] == int(head) + 1:
                    self.direct_correct += 1.0
                    self.undirect_correct += 1.0
                    if int(head) + 1 == 0:
                        self.correct_root += 1.0

                elif gold[int(head)] == int(child) + 1:
                    self.undirect_correct += 1.0

                self.total += 1.0

    def __repr__(self):
        return "UDAS: {}, UUAS:{}, root:{} ".format(
            self.score,
            self.undirect_correct / self.total,
            self.correct_root / self.total_sentence,
        )


class LossMetric(Metric):
    def __init__(self, eps=1e-8):
        super(Metric, self).__init__()
        self.eps = eps
        self.total = 0.0
        self.total_likelihood = 0.0
        self.total_kl = 0.0
        self.calling_time = 0

    def __call__(self, likelihood):
        self.calling_time += 1
        self.total += likelihood.shape[0]
        self.total_likelihood += likelihood.detach_().sum()

    @property
    def avg_loss(self):
        return self.total_likelihood / self.total

    def __repr__(self):
        return "avg likelihood: {} kl: {}, total likelihood:{}, n:{}".format(
            self.avg_likelihood, self.avg_kl, self.total_likelihood, self.total
        )

    @property
    def score(self):
        return (self.avg_likelihood + self.avg_kl).item()


# def


class MetricAccumulator(Metric):
    def __init__(self, eps=1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.total = 0.0
        self.cnt = 0.0

    def __call__(self, value):
        self.total += value
        self.cnt += 1

    @property
    def average(self):
        return self.total / self.cnt


class LikelihoodMetric(Metric):
    def __init__(self, eps=1e-8):
        super(Metric, self).__init__()
        self.eps = eps
        self.total = 0.0
        self.total_likelihood = 0.0
        self.total_word = 0

    @property
    def score(self):
        return self.avg_likelihood

    def __call__(self, likelihood, lens):

        self.total += likelihood.shape[0]
        self.total_likelihood += likelihood.detach_().sum()
        # Follow Yoon Kim
        self.total_word += lens.sum() + lens.shape[0]

    @property
    def avg_likelihood(self):
        return self.total_likelihood / self.total

    @property
    def perplexity(self):
        return (-self.total_likelihood / self.total_word).exp()

    def __repr__(self):
        return "avg likelihood: {}, perp. :{}".format(
            self.avg_likelihood, self.perplexity
        )
