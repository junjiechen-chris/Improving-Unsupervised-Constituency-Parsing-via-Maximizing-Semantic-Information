from nltk.lm import Vocabulary
from tqdm import tqdm
from easydict import EasyDict as edict
from collections import Counter
from itertools import chain, repeat


class MyVocab:
    def __init__(
        self,
        unk_cutoff=3,
        tokenizer=None,
        max_subtoken_len=None,
        max_size=None,
        override_special_tokens={},
    ) -> None:
        self.word_list = []
        self.max_size = max_size
        self.unk_cutoff = unk_cutoff

        self.special_tokens = edict(
            {
                "pad": "<PAD>",
                "unk": "<UNK>",
                **override_special_tokens,
            }
        )

        print(self.special_tokens)

        self.vocab = Vocabulary(
            unk_cutoff=unk_cutoff, unk_label=self.special_tokens.unk
        )

        self.word_tokens_offset = 20

        self.cache = []
        # self.vocab.update([self.pad_token] * 10)
        self.max_subtoken_len = max_subtoken_len
        if max_subtoken_len is not None:
            assert (
                tokenizer is not None
            ), "tokenizer must be provided if max_subtoken_len is not None"
            self.tokenizer = tokenizer

    def count(self, payload, field="word"):
        for i in tqdm(payload[field]):
            self.manual_count(i)
        self.commit_count()

    def manual_count(self, words, commit_min_count=500):
        # for i in words:
        self.cache.extend(words)
        if len(self.cache) >= commit_min_count:
            self.commit_count()

    def commit_count(self):
        if self.max_subtoken_len is not None:
            self.cache = [
                w
                for w in self.cache
                if len(self.tokenizer.tokenize(w)) <= self.max_subtoken_len
            ]
        self.vocab.update(self.cache)
        self.cache = []

    def compute_word2subtoken_mapping(self, tokenizer):
        assert (
            "idx2word" in self.__dict__
        ), "vocab must be initialized first, missing idx2word"
        self.idx2subtokenGroup = [
            tokenizer.convert_tokens_to_ids(tokenizer.tokenize(w))
            for w in self.idx2word
        ]
        print(self.idx2subtokenGroup[:10])
        max_subtoken_len = max([len(i) for i in self.idx2subtokenGroup])
        self.scatter_idx = [
            [
                idx
                for idx, subtokenGroup in enumerate(self.idx2subtokenGroup)
                if len(subtokenGroup) == i
            ]
            for i in range(1, max_subtoken_len + 1)
        ]
        self.scatter_subtokenGroup = [
            [
                subtokenGroup
                for idx, subtokenGroup in enumerate(self.idx2subtokenGroup)
                if len(subtokenGroup) == i
            ]
            for i in range(1, max_subtoken_len + 1)
        ]

    def compute_mapping(self):
        if self.max_size is not None:
            self.vocab = Vocabulary(Counter(
                {i: j for i, j in self.vocab.counts.most_common(min(self.max_size, len(self.vocab.counts)))}
            ))
        filtered_words = [x for x in self.vocab.counts.keys()]
        self.idx2word = (
            list(self.special_tokens.values())
            + (self.word_tokens_offset - len(self.special_tokens)) * ["<ILLEGAL>"]
            + filtered_words
        )
        # self.idx2word = [self.pad_token, self.vocab.unk_label, self.bos_token, self.eos_token,self.mask_token, self.shift_token, self.reduce_token]+sorted([x for x, count in self.vocab.counts.items() if count >= self.unk_cutoff])
        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}
        # self.pad_id = self.word2idx[self.pad_token]
        self.vocab_size = len(self.idx2word)

        self.special_ids = edict(
            {k: self.word2idx[v] for k, v in self.special_tokens.items()}
        )

    def compute_unigram_counts(self, payload, field="word"):
        # input_ids = []
        # for i in tqdm(payload[field]):
        # print(i)
        # input_ids.append(i)
        self.unigram_count = Counter(chain(*payload[field]))
        self.total_unigram_count = sum(self.unigram_count.values())

    def _index(self, words, pos=None):
        return [self.word2idx[self._wlookup(w, p)] for w, p in zip(words, pos if pos is not None else repeat(None))]

    def _lookup(self, words, pos):
        if pos is None:
            return [self.vocab.lookup(w) for w in words]
        else:
            ret = [self.vocab.lookup(w) for w in words]
            aug_ret = [w if w!= '<UNK>' else f'<UNK-{p}>' for w, p in zip(ret, pos)]
            return aug_ret
            
    def _wlookup(self, word, pos):
        if pos is None:
            return self.vocab.lookup(word) 
        else:
            ret = self.vocab.lookup(word)
            if ret == '<UNK>':
                return f'<UNK-{pos}>'
            else:
                return ret

    def index_dataset(self, dataset, source_label="word_form", target_label="word"):
        dataset.add_field(target_label, self._index, source_label)

    def _manual_index(self, payload, source_field, target_field):
        # for i in payload[source_field]:
        payload[target_field] = [self._index(i) for i in payload[source_field]]
        # payload.add_field(target_field, self._index, source_field)

    def convert_ids_to_words(self, ids):
        return [self.idx2word[id] for id in ids]
