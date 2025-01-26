import itertools


from parsing_by_maxseminfo.utils.utils import convert_ptb_words_and_spans_to_spacy_tokens_and_spans
from typing import List, Union, Dict, Any

from parsing_by_maxseminfo.parser.helper.utils_spanoverlap import removing_punctuations

# from fastNLP.core import sampler
from .data_module import ByLengthSamplerV1, DataModule, ByLengthSampler
import pickle
from easydict import EasyDict as edict
from .dataset import MyDataset, MyDataIter
import numpy as np
import random
import torch


def test_span_contained(s_test, s_ref):
    return any([s_test[0] >= s[0] and s_test[-1] <= s[-1] for s in s_ref])


def clean_word(words):
    import re

    def clean_number(w):
        new_w = re.sub("[0-9]{1,}([,.]?[0-9]*)*", "N", w)
        return new_w

    return [clean_number(word.lower()) for word in words]


lang_punctpos_map = {
    "english": lambda p: p
    not in ["``", "''", ":", "-RRB-", ",", ".", "-LRB-", "#", "$"],
    "german": lambda p: not (len(p) > 1 and p.startswith("$")),
    "polish": lambda p: not (len(p) > 1 and p.startswith("interp")),
    "french": lambda p: not (len(p) > 1 and p.startswith("PONCT")),
    "basque": lambda p: not (
        len(p) > 1 and any([p.startswith(i) for i in ["PUNT", "BEREIZ"]])
    ),
    "hungarian": lambda p: not (len(p) > 1 and p.startswith("PUNC")),
    "hebrew": lambda p: not (
        len(p) > 1
        and any(
            [
                p.startswith(i)
                for i in [
                    "yyDOT",
                    "yyCM",
                    "yyLRB",
                    "yyRRB",
                    "yyQM",
                    "yyDASH",
                ]
            ]
        )
    ),
    "korean": lambda p: not (
        len(p) > 1 and any([p.startswith(i) for i in ["sf", "sp", "su", "sl", "sr"]])
    ),
    "swedish": lambda p: not (
        len(p) > 1
        and (p.startswith("MAD") or p.startswith("PAD") or p.startswith("MID"))
    ),
    "chinese": lambda p: not (len(p) > 1 and (p.startswith("PU"))),
}


class PASCtrlPCFGDataset(MyDataset):
    def __init__(self, **kwargs):
        self.payload = {**kwargs}

    @classmethod
    def from_pickle(cls, pickle_file, langstr):
        payload = {
            "langstr": langstr,
            **pickle_file,
        }
        # print(payload)
        return cls(**payload)

    @classmethod
    def from_payload(cls, payload):
        return cls(**payload)


def test_span_conflict(s1, s2):
    i, j = s1
    p, q = s2
    # print(s1, s2)
    return (i < p and j > p and j < q) or (i > p and i < q and j > q)


def normalizing_permuted_en(x):
    return (
        x.replace("n't", " n't")
        .replace("'s", " 's")
        .replace("%", " %")
        .replace("$", "$ ")
        .replace("(", " -LRB- ")
        .replace(")", " -RRB- ")
        .replace("'m", " 'm")
        .replace("n't", "not")
    )


def normalizing_permuted_fr(x):
    return (
        x.replace("l'", "l' ")
        .replace("d'", "d' ")
        .replace("n'", "n' ")
        # .replace("(", " -LRB- ")
        # .replace(")", " -RRB- ")
    )  # .replace('l\'', 'les').replace('d\'', 'das')


def normalizing_permuted(x, lang_str):
    if lang_str == "french":
        # return x
        return normalizing_permuted_fr(x)
    # elif lang_str == "english":
    #     return normalizing_permuted_en(x)
    else:
        return x


def clean_punct_from_trees(dst, save_field="cleaned_gold_tree"):

    gold_trees = dst.payload["gold_tree"]
    pos_array = dst.payload["pos"]
    token_idx2word_idx = [
        np.cumsum([lang_punctpos_map[dst.payload["langstr"]](p) for p in pos])
        for pos in pos_array
    ]
    cleaned_gold_trees = [
        [
            (
                token_idx2word_idx[bid][s[0] - 1] if s[0] > 1 else 0,
                token_idx2word_idx[bid][s[1] - 1],
                s[2],
            )
            for s in spans
        ]
        for bid, spans in enumerate(gold_trees)
    ]
    dst.payload[save_field] = cleaned_gold_trees


def clean_punct_from_spacy_trees(
    dst, spacy_token_array, spacy_span_array, save_field="cleaned_gold_tree"
):

    gold_trees = spacy_span_array
    pos_array = [[t[1] for t in tokens] for tokens in spacy_token_array]
    token_idx2word_idx = [np.cumsum([p != "PUNCT" for p in pos]) for pos in pos_array]
    cleaned_gold_trees = [
        [
            (
                token_idx2word_idx[bid][s[0] - 1] if s[0] > 1 else 0,
                token_idx2word_idx[bid][s[1] - 1],
                s[2],
            )
            for s in spans
        ]
        for bid, spans in enumerate(gold_trees)
    ]
    dst.payload[save_field] = cleaned_gold_trees


def clean_punct_from_spacy_toks(
    dst, spacy_token_array, save_field="cleaned_word", langstr="english"
):
    from parser.helper.utils_spanoverlap import (
        normalizing_string,
        removing_punctuations,
    )
    from Stemmer import Stemmer

    print(dst.payload.keys())

    cleaned_form_array = [
        [
            # normalizing_string(w, stemmer)
            w.lower()
            for w, p, _, _ in token
            if p != "PUNCT"
        ]
        for token in spacy_token_array
    ]  # clean form of punctuations

    lemma_array = [
        [
            # normalizing_string(w, stemmer)
            l.lower()
            for _, p, _, l in token
            if p != "PUNCT"
        ]
        for token in spacy_token_array
    ]  # clean form of punctuations

    len_array = [len(token) for token in cleaned_form_array]

    dst.payload[f"{save_field}-withnum"] = cleaned_form_array

    cleaned_form_array = [
        clean_word(form) for form in cleaned_form_array
    ]  # clean form of numbers

    dst.payload[save_field] = cleaned_form_array
    dst.payload["trg_lemma_array"] = lemma_array
    dst.payload["seqlen"] = len_array

def clean_punct_from_pos(dst, save_field="cleaned_pos_array", langstr="english"):
    pos_array = dst.payload["pos"]
    cleaned_pos_array = [
        [p for p in pos if lang_punctpos_map[dst.payload["langstr"]](p)]
        for pos in pos_array
    ]
    dst.payload[save_field] = cleaned_pos_array


def clean_punct_from_words(dst, save_field="cleaned_word", langstr="english"):
    from parser.helper.utils_spanoverlap import (
        normalizing_string,
        removing_punctuations,
    )
    from Stemmer import Stemmer

    print(dst.payload.keys())

    # stemmer = Stemmer(dst.payload["langstr"])
    form_array = dst.payload["word"]
    pos_array = dst.payload["pos"]
    len_array = [
        sum([lang_punctpos_map[dst.payload["langstr"]](p) for p in pos])
        for pos in pos_array
    ]

    cleaned_form_array = [
        [
            # normalizing_string(w, stemmer)
            w.lower()
            for w, p in zip(form, pos)
            if lang_punctpos_map[dst.payload["langstr"]](p)
        ]
        for form, pos in zip(form_array, pos_array)
    ]  # clean form of punctuations

    cleaned_pos_array = [
        [p for p in pos if lang_punctpos_map[dst.payload["langstr"]](p)]
        for pos in pos_array
    ]

    dst.payload[f"{save_field}-withnum"] = cleaned_form_array

    cleaned_form_array = [
        clean_word(form) for form in cleaned_form_array
    ]  # clean form of numbers

    dst.payload[save_field] = cleaned_form_array
    dst.payload["seqlen"] = len_array
    dst.payload['cleaned_pos_array'] = cleaned_pos_array


def extract_spacy_tokenization_and_span(dst, langstr="english"):
    langstr2spacystr = {
        "english": "en_core_web_sm",
        "german": "de_core_news_sm",
        "french": "fr_core_news_sm",
        "chinese": "zh_core_web_sm",
    }
    from Stemmer import Stemmer

    print(dst.payload.keys())
    print("loading spacy for ", langstr)

    # stemmer = Stemmer(dst.payload["langstr"])
    form_array = dst.payload["word"]
    span_array = dst.payload["gold_tree"]
    # pos_array = dst.payload["pos"]
    nlp = spacy.load(langstr2spacystr[langstr])

    spacy_tok_array, spacy_span_array, spacy_w2t_mapping = [], [], []
    for form, span in zip(form_array, span_array):
        spacy_tok, spacy_span, w2t_mapping = (
            convert_ptb_words_and_spans_to_spacy_tokens_and_spans(form, span, nlp)
        )
        spacy_tok_array.append(spacy_tok)
        spacy_span_array.append(spacy_span)
        spacy_w2t_mapping.append(w2t_mapping)

    return spacy_tok_array, spacy_span_array, spacy_w2t_mapping


import spacy


def sample_set_processor(sample_list, langstr):
    from tqdm import tqdm

    langstr2spacystr = {
        "english": "en_core_web_sm",
        "german": "de_core_news_sm",
        "french": "fr_core_news_sm",
        "chinese": "zh_core_web_sm",
    }
    print("loading spacy for ", langstr)
    spacy_model = spacy.load(langstr2spacystr[langstr])

    def replace_parentheses(words):
        return words.replace("-lrb-", "(").replace("-rrb-", ")")

    out = [
        [
            [
                (token.text, token.lemma_.lower())
                for token in spacy_model(replace_parentheses(sample))
                if token.pos_ != "PUNCT"
            ]
            for sample in sample_set
        ]
        for sample_set in tqdm(sample_list)
    ]
    # print(out[0])
    return out


# class SpacyTokenizer


def clean_and_split_pas_samples_spacy(
    dst,
    save_field="tokenized_pas_samples",
    langstr="english",
    pas_subsample_count=10000,
    distinct_prompt=False,
):
    from parser.helper.utils_spanoverlap import (
        normalizing_string,
        removing_punctuations,
    )
    from tqdm import tqdm
    import multiprocessing
    from functools import partial

    assert dst.payload["langstr"] in [
        "english",
        "german",
        "french",
        "chinese",
    ], f"only {dst.payload['langstr']} is implemented currently"
    samples = dst.payload["pas_sample"]
    prompt_keys = dst.payload["pas_prompt_key"]

    # exclude the tense_gd_en_ew prompt
    samples = [
        [
            sample
            for sample, prompt_key in zip(sample_set, prompt_set)
            if prompt_key != "tense_gd_en_ew"
        ]
        for sample_set, prompt_set in zip(samples, prompt_keys)
    ]

    if distinct_prompt:
        samples = [list(set(sst)) for sst in samples]

    for sample_set in samples:
        random.shuffle(sample_set)
    samples = [sample_set[:pas_subsample_count] for sample_set in samples]

    # with multiprocessing.Manager() as manager:

    # print(shared_spacy_model)

    sample_set_processor_map = partial(sample_set_processor, langstr=langstr)

    num_workers = 16
    chunk_size = int(len(samples) / num_workers) + 1
    split_samples = [
        samples[i * chunk_size : (i + 1) * chunk_size] for i in range(num_workers)
    ]
    with multiprocessing.Pool(num_workers) as p:
        cleaned_samples = p.map(sample_set_processor_map, split_samples)
    cleaned_samples_pack = [
        i for worker_output in cleaned_samples for i in worker_output
    ]

    dst.payload[f"{save_field}-withnum"] = cleaned_samples_pack

    cleaned_samples = [
        [clean_word([w[0] for w in words]) for words in sample_set]
        for sample_set in cleaned_samples_pack
    ]
    # print(cleaned_samples_pack[0])

    # for sample_set in cleaned_samples_pack:
    #     for words in sample_set:
    #         assert len(words) == 2, f"words: {words}"
    lemma_array = [
        [[w[1] for w in words] for words in sample_set]
        for sample_set in cleaned_samples_pack
    ]

    cleaned_samples_concat = list(itertools.chain(*cleaned_samples))

    dst.payload[save_field] = cleaned_samples
    dst.payload["pas_lemma_array"] = lemma_array
    dst.payload["flatten_pas_sample"] = cleaned_samples_concat


def clean_and_split_pas_samples(dst, save_field="tokenized_pas_samples"):
    from parser.helper.utils_spanoverlap import (
        normalizing_string,
        removing_punctuations,
    )
    from Stemmer import Stemmer
    from nltk import word_tokenize
    import itertools

    print("doing for ", dst.payload["langstr"])
    # assert dst.payload["langstr"] == "english", "only english is implemented currently"
    if dst.payload["langstr"] == 'chinese':
        stemmer = IdentityStemmer()
    else:
        stemmer = Stemmer(dst.payload["langstr"])
    samples = dst.payload["pas_sample"]

    if dst.payload["langstr"] == "chinese":
        word_tokenize = lambda x, language: [w for w in x]
    def wrapped_word_tokenzie(sent, language):
        try:
            return word_tokenize(sent, language)
        except:
            print(f"ERROR!: {sent}")
            return ""

    cleaned_samples = [
        [
            wrapped_word_tokenzie(
                # removing_punctuations(normalizing_permuted(i, dst.payload["langstr"])), language=dst.payload["langstr"]
                normalizing_permuted(i, dst.payload["langstr"]),
                language=dst.payload["langstr"],
            )
            for i in sample_set
        ]
        for sample_set in samples
    ]
    cleaned_samples = [
        [[removing_punctuations(w).strip() for w in i] for i in sample_set]
        for sample_set in cleaned_samples
    ]
    cleaned_samples = [
        [[w for w in i if len(w) > 0] for i in sample_set]
        for sample_set in cleaned_samples
    ]

    dst.payload[f"{save_field}-withnum"] = cleaned_samples

    cleaned_samples = [
        [clean_word(words) for words in sample_set] for sample_set in cleaned_samples
    ]

    cleaned_samples_concat = list(itertools.chain(*cleaned_samples))

    dst.payload[save_field] = cleaned_samples
    dst.payload["flatten_pas_sample"] = cleaned_samples_concat


def convert_discontinuous_w2t_mapping_to_continuous(w2t_mapping):
    cum_list = np.cumsum([len(g) for g in w2t_mapping])
    new_w2t_mapping = [
        [cum_list[i - 1] + j if i > 0 else j for j in range(len(g))]
        for i, g in enumerate(w2t_mapping)
    ]
    return new_w2t_mapping


def clean_punct_from_w2tmappings(
    dst, w2t_mapping, spacy_token_array, save_field="cleaned_w2t_mapping"
):
    # print(spacy_token_array[0])
    w2t_mapping = [
        [
            [
                idx for idx in g if spacy_token_array[mid][idx][1] != "PUNCT"
            ]  # remove the end of sentence & punctuation
            for g in mapping[:-1]
        ]
        for mid, mapping in enumerate(w2t_mapping)
    ]
    w2t_mapping = [[g for g in mapping if len(g) > 0] for mapping in w2t_mapping]
    w2t_mapping = [
        convert_discontinuous_w2t_mapping_to_continuous(mapping)
        for mapping in w2t_mapping
    ]
    w2t_mapping = [
        mapping + [[mapping[-1][-1] + 1]] if len(mapping) > 0 else mapping
        for mapping in w2t_mapping
    ]  # add the end of the sentence
    dst.payload[save_field] = w2t_mapping


class IdentityStemmer:
    def stemWord(self, x):
        return x


class DataModuleForPASCtrlPCFG(DataModule):
    def __init__(
        self,
        hparams,
        langstr,
        use_cache=True,
        max_size=10000000,
        merge_pas_data=False,
        flag_use_spacy_preprocessing=False,
        flag_spanoverlap_match_char=False,
        flag_use_spacy_for_treebank=False,
        pas_subsample=10000,
        flag_compute_relative_frequency=False,
        distinct_prompt=False,
        flag_use_pos_unks=False,
    ):
        self.langstr = langstr
        self.use_cache = use_cache
        self.max_size = max_size
        self.merge_pas_data = merge_pas_data
        self.flag_use_spacy_preprocessing = flag_use_spacy_preprocessing
        self.flag_spanoverlap_match_char = flag_spanoverlap_match_char
        self.flag_use_spacy_for_treebank = flag_use_spacy_for_treebank
        self.pas_subsample = pas_subsample
        self.flag_compute_relative_frequency = flag_compute_relative_frequency
        self.distinct_prompt = distinct_prompt
        self.flag_use_pos_unks = flag_use_pos_unks
        super().__init__(hparams)

        print(f"Preparing datasets with {self.pas_subsample} PAS samples")

    def setup(self):
        from parsing_by_maxseminfo.parser.helper.vocab import MyVocab

        assert (
            not self.flag_use_spacy_for_treebank != self.flag_use_spacy_preprocessing
        ), "spacy mode must be enabled or disabled for both treebank and pas samples"

        data = self.hparams.data
        print(self.langstr)

        if not self.use_cache:
            self.word_vocab = MyVocab(unk_cutoff=2, max_size=self.max_size)

            has_ood = hasattr(data, "test_ood_file")
            print("Current dataset contains OOD data:", has_ood)
            train_data = pickle.load(open(data.train_file, "rb"))
            test_data = pickle.load(open(data.test_file, "rb"))
            val_data = pickle.load(open(data.val_file, "rb"))
            if has_ood:
                test_ood_data = pickle.load(open(data.test_ood_file, "rb"))
            # print(val_data["pas_sample"][:10])
            self.train_dataset = PASCtrlPCFGDataset.from_pickle(
                train_data, self.langstr
            )
            self.val_dataset = PASCtrlPCFGDataset.from_pickle(val_data, self.langstr)
            self.test_dataset = PASCtrlPCFGDataset.from_pickle(test_data, self.langstr)
            if has_ood:
                self.test_ood_dataset = PASCtrlPCFGDataset.from_pickle(
                    test_ood_data, self.langstr
                )

            if not self.flag_use_spacy_for_treebank:
                # using the treebank tokenization
                clean_punct_from_words(
                    self.train_dataset, save_field="cleaned_word_form"
                )
                clean_punct_from_words(self.val_dataset, save_field="cleaned_word_form")
                clean_punct_from_words(
                    self.test_dataset, save_field="cleaned_word_form"
                )
                if has_ood:
                    clean_punct_from_words(
                        self.test_ood_dataset, save_field="cleaned_word_form"
                    )

                print("finished corpus punct normalization")

                clean_punct_from_trees(
                    self.train_dataset, save_field="cleaned_gold_tree"
                )
                clean_punct_from_trees(self.val_dataset, save_field="cleaned_gold_tree")
                clean_punct_from_trees(
                    self.test_dataset, save_field="cleaned_gold_tree"
                )
                if has_ood:
                    clean_punct_from_trees(
                        self.test_ood_dataset, save_field="cleaned_gold_tree"
                    )
                print("finished gold tree punct normalization")
            else:
                train_toks, train_spans, train_w2t = (
                    extract_spacy_tokenization_and_span(
                        self.train_dataset, langstr=self.langstr
                    )
                )
                val_toks, val_spans, val_w2t = extract_spacy_tokenization_and_span(
                    self.val_dataset, langstr=self.langstr
                )
                test_toks, test_spans, test_w2t = extract_spacy_tokenization_and_span(
                    self.test_dataset, langstr=self.langstr
                )
                if has_ood:
                    test_ood_toks, test_ood_spans, test_ood_w2t = (
                        extract_spacy_tokenization_and_span(
                            self.test_ood_dataset, langstr=self.langstr
                        )
                    )

                clean_punct_from_w2tmappings(self.train_dataset, train_w2t, train_toks)
                clean_punct_from_w2tmappings(self.val_dataset, val_w2t, val_toks)
                clean_punct_from_w2tmappings(self.test_dataset, test_w2t, test_toks)
                if has_ood:
                    clean_punct_from_w2tmappings(
                        self.test_ood_dataset, test_ood_w2t, test_ood_toks
                    )

                clean_punct_from_spacy_toks(
                    self.train_dataset, train_toks, save_field="cleaned_word_form"
                )
                clean_punct_from_spacy_toks(
                    self.val_dataset, val_toks, save_field="cleaned_word_form"
                )
                clean_punct_from_spacy_toks(
                    self.test_dataset, test_toks, save_field="cleaned_word_form"
                )
                if has_ood:
                    clean_punct_from_spacy_toks(
                        self.test_ood_dataset,
                        test_ood_toks,
                        save_field="cleaned_word_form",
                    )

                clean_punct_from_spacy_trees(
                    self.train_dataset,
                    train_toks,
                    train_spans,
                    save_field="cleaned_gold_tree",
                )
                clean_punct_from_spacy_trees(
                    self.val_dataset,
                    val_toks,
                    val_spans,
                    save_field="cleaned_gold_tree",
                )
                clean_punct_from_spacy_trees(
                    self.test_dataset,
                    test_toks,
                    test_spans,
                    save_field="cleaned_gold_tree",
                )
                if has_ood:
                    clean_punct_from_spacy_trees(
                        self.test_ood_dataset,
                        test_ood_toks,
                        test_ood_spans,
                        save_field="cleaned_gold_tree",
                    )
                print("finished corpus punct normalization for words and trees")

            if self.flag_use_spacy_preprocessing:
                clean_and_split_pas_samples_spacy(
                    self.train_dataset,
                    save_field="cleaned_pas_sample",
                    langstr=self.langstr,
                    pas_subsample_count=self.pas_subsample,
                    distinct_prompt=self.distinct_prompt,
                )
                clean_and_split_pas_samples_spacy(
                    self.val_dataset,
                    save_field="cleaned_pas_sample",
                    langstr=self.langstr,
                    pas_subsample_count=self.pas_subsample,
                    distinct_prompt=self.distinct_prompt,
                )
                clean_and_split_pas_samples_spacy(
                    self.test_dataset,
                    save_field="cleaned_pas_sample",
                    langstr=self.langstr,
                    distinct_prompt=self.distinct_prompt,
                    # pas_subsample_count=self.pas_subsample
                )
                if has_ood:
                    clean_and_split_pas_samples_spacy(
                        self.test_ood_dataset,
                        save_field="cleaned_pas_sample",
                        langstr=self.langstr,
                        distinct_prompt=self.distinct_prompt,
                        # pas_subsample_count=self.pas_subsample
                    )
            else:
                clean_and_split_pas_samples(
                    self.train_dataset, save_field="cleaned_pas_sample"
                )
                clean_and_split_pas_samples(
                    self.val_dataset, save_field="cleaned_pas_sample"
                )
                clean_and_split_pas_samples(
                    self.test_dataset, save_field="cleaned_pas_sample"
                )
                if has_ood:
                    clean_and_split_pas_samples(
                        self.test_ood_dataset, save_field="cleaned_pas_sample"
                    )

            print("finished pas samples normalization")

            self.word_vocab.count(self.train_dataset.payload, field="cleaned_word_form")
            if self.flag_use_spacy_preprocessing and self.flag_use_spacy_for_treebank:
                self.word_vocab.count(
                    self.train_dataset.payload, field="flatten_pas_sample"
                )
            print("computing mapping")
            self.word_vocab.compute_mapping()

            num_workers = 32

            c_spacy = (
                self.flag_use_spacy_preprocessing and self.flag_use_spacy_for_treebank
            )
            self.prep_dataset_with_spanoverlap_seq(
                self.train_dataset,
                self.langstr,
                pas_subsample_count=self.pas_subsample,
                flag_compute_relative_frequency=self.flag_compute_relative_frequency,
                flag_use_spacy_preprocessing=c_spacy,
                workers=num_workers,
            )
            self.prep_dataset_with_spanoverlap_seq(
                self.val_dataset,
                self.langstr,
                pas_subsample_count=self.pas_subsample,
                flag_compute_relative_frequency=self.flag_compute_relative_frequency,
                flag_use_spacy_preprocessing=c_spacy,
                workers=num_workers,
            )
            self.prep_dataset_with_spanoverlap_seq(
                self.test_dataset,
                self.langstr,
                pas_subsample_count=self.pas_subsample,
                flag_compute_relative_frequency=self.flag_compute_relative_frequency,
                flag_use_spacy_preprocessing=c_spacy,
                workers=num_workers,
            )

            self.save_processed(subsample=self.pas_subsample)
        else:
            import pathlib

            data_path = pathlib.Path(data.train_file).parent.resolve()
            has_ood = hasattr(data, "test_ood_file")

            subsample_str = (
                f".ss{self.pas_subsample}" if self.pas_subsample < 10000 else ""
            )
            print("loading from ", data_path)
            self.train_dataset = pickle.load(
                open(f"{data.train_file}.processed{subsample_str}", "rb")
            )
            self.test_dataset = pickle.load(
                open(f"{data.test_file}.processed{subsample_str}", "rb")
            )
            self.val_dataset = pickle.load(
                open(f"{data.val_file}.processed{subsample_str}", "rb")
            )
            if has_ood:
                self.test_ood_dataset = pickle.load(
                    open(f"{data.test_ood_file}.processed{subsample_str}", "rb")
                )
            self.word_vocab = pickle.load(open(f"{data_path}/vocab.pkl", "rb"))

            if self.flag_use_pos_unks:
                assert self.langstr == "german", "only german is allowed to use pos unks"
                # pos_set = set(sum(self.train_dataset.payload["pos"], []))
                posset = set([w.split('##')[0].split('-')[0] for s in self.train_dataset.payload["pos"] for w in s])
                pos_unk = [f'<UNK-{w}>' for w in posset]
                self.word_vocab.idx2word.extend(pos_unk)
                word_vocab_size = self.word_vocab.vocab_size
                self.word_vocab.word2idx.update({w: i+word_vocab_size for i, w in enumerate(pos_unk)})
                self.word_vocab.vocab_size = len(self.word_vocab.idx2word)
                for idx, w in enumerate(self.word_vocab.idx2word):
                    if w == '<ILLEGAL>': continue
                    backmap_idx = self.word_vocab.word2idx[w]
                    assert idx == backmap_idx, f"{idx} != {backmap_idx}, {w}"

                clean_punct_from_pos(self.train_dataset)
                clean_punct_from_pos(self.val_dataset)
                clean_punct_from_pos(self.test_dataset)





        if self.merge_pas_data:
            self.merge_pas_original(self.train_dataset)
        # self.word_vocab.index_dataset()

    @staticmethod
    def f_worker(word_forms, pas_samples, langstr, match_character_only=False):
        from tqdm import tqdm
        from Stemmer import Stemmer

        scorer = SpanScorer()
        if langstr == "chinese":
            stemmer = IdentityStemmer()
        else:
            stemmer = Stemmer(langstr)
        spanoverlap_score_array = []
        hit_counts_array = []
        for trg, trg_pas in tqdm(zip(word_forms, pas_samples)):
            scores = np.zeros((len(trg) + 1, len(trg) + 1))
            hit_counts = []
            for sample in trg_pas:
                hitscore = scorer.score_by_longest_matches(
                    trg,
                    sample,
                    stemmer,
                    flag_print=False,
                    match_character_only=match_character_only,
                )
                hits = hitscore.nonzero()
                hit_counts.append(len(hits[0]))
                for i, j, l, r in zip(hits[0], hits[1], hits[2], hits[3]):
                    scores[i, j] += 1
            spanoverlap_score_array.append(scores / len(trg_pas))
            hit_counts_array.append(hit_counts)
        return spanoverlap_score_array, hit_counts_array

    @staticmethod
    def prep_dataset_with_spanoverlap_score(
        dst, langstr, match_character_only=False, workers=14, pas_subsample_count=None
    ):
        import multiprocessing
        from itertools import repeat

        assert pas_subsample_count is not None, "pas_subsample_count must be specified"

        # from tqdm import tqdm

        pas_samples = dst.payload["pas_lemma_array"]
        word_forms = dst.payload["trg_lemma_array"]

        pas_samples = [
            random.sample(pas, k=min(pas_subsample_count, len(pas)))
            for pas in pas_samples
        ]

        chunk_size = int(len(word_forms) / workers) + 2
        chunked_word_forms = [
            word_forms[i * chunk_size : (i + 1) * chunk_size] for i in range(workers)
        ]
        chunked_pas_samples = [
            pas_samples[i * chunk_size : (i + 1) * chunk_size] for i in range(workers)
        ]
        print(f"do multiprocessing spanoverlap computation with {workers} workers")
        spanoverlap_score_array = []
        hit_counts_array = []
        with multiprocessing.Pool(workers) as p:
            outputs = p.starmap(
                DataModuleForPASCtrlPCFG.f_worker,
                zip(
                    chunked_word_forms,
                    chunked_pas_samples,
                    repeat(langstr),
                    repeat(match_character_only),
                ),
            )
        for output in outputs:
            spanoverlap_score_array.extend(output[0])
            hit_counts_array.extend(output[1])
        # spanoverlap_score_array = sum(spanoverlap_score_array, [])
        # spanoverlap_score_array, hit_counts_array = zip(*spanoverlap_score_array)

        dst.payload["spanoverlap_score"] = spanoverlap_score_array
        dst.payload["pas_hit_counts"] = hit_counts_array

    @staticmethod
    def f_worker_soseq(
        word_forms, pas_samples, langstr, flag_compute_relative_frequency
    ):
        from ipynb.utils import so_accumulation
        from tqdm import tqdm
        from Stemmer import Stemmer

        if langstr == "chinese":
            stemmer = IdentityStemmer()
        else:
            stemmer = Stemmer(langstr)
        trg_spanoverlap_score_array = []
        pas_spanoverlap_score_array = []
        # hit_counts_array = []
        for trg, trg_pas in tqdm(zip(word_forms, pas_samples)):
            form_array = [trg] + trg_pas
            # print(form_array)
            # input()
            so_scores = so_accumulation(
                form_array,
                stemmer,
                flag_compute_relative_frequency=flag_compute_relative_frequency,
            )
            trg_spanoverlap_score_array.append(so_scores[0])
            pas_spanoverlap_score_array.append(so_scores[1:])
        return trg_spanoverlap_score_array, pas_spanoverlap_score_array

    @staticmethod
    def f_worker_soseq_tbtok(
        word_forms, pas_samples, langstr, flag_compute_relative_frequency, ref_word_forms
    ):
        from ipynb.utils import so_accumulation_tbtok
        from tqdm import tqdm
        from Stemmer import Stemmer

        if langstr == "chinese":
            stemmer = IdentityStemmer()
        else:
            stemmer = Stemmer(langstr)
        sentence_joiner = " " if langstr != "chinese" else ""

        ref_word_forms = [[stemmer.stemWord(removing_punctuations(w).strip()).lower() for w in seq] for seq in ref_word_forms]
        ref_word_forms = [sentence_joiner.join([w for w in seq if len(w) > 0]) for seq in ref_word_forms]

        trg_spanoverlap_score_array = []
        pas_spanoverlap_score_array = []
        # hit_counts_array = []
        for trg, trg_pas in tqdm(zip(word_forms, pas_samples)):
            form_array = [trg] + trg_pas
            # print(form_array)
            # input()
            so_scores = so_accumulation_tbtok(
                form_array,
                stemmer,
                flag_compute_relative_frequency=flag_compute_relative_frequency,
                corpus=ref_word_forms,
                langstr=langstr
            )
            trg_spanoverlap_score_array.append(so_scores[0])
            pas_spanoverlap_score_array.append(so_scores[1:])
        return trg_spanoverlap_score_array, pas_spanoverlap_score_array

    @staticmethod
    def prep_dataset_with_spanoverlap_seq(
        dst,
        langstr,
        workers=32,
        pas_subsample_count=None,
        flag_compute_relative_frequency=False,
        flag_use_spacy_preprocessing=False,
    ):
        import multiprocessing
        from itertools import repeat

        assert pas_subsample_count is not None, "pas_subsample_count must be specified"

        print(f"compute relative frequency: {flag_compute_relative_frequency}")
        print(f"using spacy preprocessing: {flag_use_spacy_preprocessing}")

        # from tqdm import tqdm

        word_form_key = (
            "trg_lemma_array" if flag_use_spacy_preprocessing else "cleaned_word_form-withnum"
        )
        pas_sample_key = (
            "pas_lemma_array" if flag_use_spacy_preprocessing else "cleaned_pas_sample-withnum"
        )

        pas_samples = dst.payload[pas_sample_key]
        word_forms = dst.payload[word_form_key]
        ref_word_forms = dst.payload["cleaned_word_form"]
        assert len(word_forms) == len(ref_word_forms)
        assert all([len(w) == len(r) for w, r in zip(word_forms, ref_word_forms)])
        corpus = [w for w, cp in zip(word_forms, pas_samples)]

        # pas_samples = [random.sample(pas, k=min(pas_subsample_count, len(pas))) for pas in pas_samples]

        chunk_size = int(len(word_forms) / workers) + 2
        chunked_word_forms = [
            word_forms[i * chunk_size : (i + 1) * chunk_size] for i in range(workers)
        ]
        chunked_pas_samples = [
            pas_samples[i * chunk_size : (i + 1) * chunk_size] for i in range(workers)
        ]
        print(
            f"do multiprocessing spanoverlap computation over the PAS-equivalent set with {workers} workers"
        )
        trg_spanoverlap_score_array = []
        pas_spanoverlap_score_array = []

        with multiprocessing.Pool(workers) as p:
            outputs = p.starmap(
                (
                    DataModuleForPASCtrlPCFG.f_worker_soseq
                    if flag_use_spacy_preprocessing
                    else DataModuleForPASCtrlPCFG.f_worker_soseq_tbtok
                ),
                zip(
                    chunked_word_forms,
                    chunked_pas_samples,
                    repeat(langstr),
                    repeat(flag_compute_relative_frequency),
                    repeat(corpus)
                ),
            )
        for output in outputs:
            trg_spanoverlap_score_array.extend(output[0])
            pas_spanoverlap_score_array.extend(output[1])
            # hit_counts_array.extend(output[1])
        # spanoverlap_score_array = sum(spanoverlap_score_array, [])
        # spanoverlap_score_array, hit_counts_array = zip(*spanoverlap_score_array)

        dst.payload["trg_spanoverlap_score"] = trg_spanoverlap_score_array
        dst.payload["pas_spanoverlap_score"] = pas_spanoverlap_score_array

    @staticmethod
    def merge_pas_original(dst):
        orig = dst.payload["cleaned_word_form"]
        pas = dst.payload["cleaned_pas_sample"]
        merged = list(itertools.chain(*(pas + [orig])))
        merged_len = [len(i) for i in merged]
        dst.payload["cleaned_word_form"] = merged
        dst.payload["seqlen"] = merged_len

    def save_processed(self, subsample):
        import pathlib

        subsample_str = f".ss{subsample}" if subsample < 10000 else ""

        data = self.hparams.data
        data_path = pathlib.Path(data.train_file).parent.resolve()
        with open(f"{data.train_file}.processed{subsample_str}", "wb") as f:
            pickle.dump(self.train_dataset, f)
        with open(f"{data.val_file}.processed{subsample_str}", "wb") as f:
            pickle.dump(self.val_dataset, f)
        with open(f"{data.test_file}.processed{subsample_str}", "wb") as f:
            pickle.dump(self.test_dataset, f)
        if subsample == 10000:
            # do not alter vocab if pas subsampling is applied
            with open(f"{data_path}/vocab.pkl", "wb") as f:
                pickle.dump(self.word_vocab, f)
        if hasattr(data, "test_ood_file"):
            with open(f"{data.test_ood_file}.processed{subsample_str}", "wb") as f:
                pickle.dump(self.test_ood_dataset, f)
        print("saved!")

    def dev_dataloader(
        self,
        langstr,
        device="cuda",
        max_len=40,
        min_len=3,
        prompt_type="anything",
        flag_use_simlen_pas_data=False,
        max_pasdata_lendiff=0,
        pair_pas_with_target=False,
        pas_subsample_count=10000,
    ):
        args = self.hparams.test
        val_dataset = self.val_dataset.drop(
            lambda x: x > max_len or x < min_len, payload_label="seqlen"
        )
        val_sampler = ByLengthSampler(
            dataset=val_dataset, batch_size=args.batch_size, seq_len_label="seqlen"
        )

        params = {
            "payload": val_dataset.payload,
            "sampler": val_sampler,
            "device": device,
            "mode": "eval",
            "word_vocab": self.word_vocab,
            "use_simlen_pasdata": flag_use_simlen_pas_data,
            "max_pasdata_lendiff": max_pasdata_lendiff,
            "resample_target_data": False,
            "pair_pas_with_target": pair_pas_with_target,
            "pas_subsample_count": pas_subsample_count,
        }
        it = PASCtrlPCFGIter(langstr=langstr, prompt_type=prompt_type, **params)
        return it

    def train_dataloader(
        self,
        langstr,
        device="cuda",
        max_len=40,
        min_len=3,
        prompt_type="anything",
        flag_use_simlen_pas_data=False,
        max_pasdata_lendiff=0,
        resample_target_data=False,
        pair_pas_with_target=False,
        supervised_mode=False,
        pas_subsample_count=10000,
    ):

        args = self.hparams.train
        dst = self.train_dataset.drop(
            lambda x: x > max_len or x < min_len, payload_label="seqlen"
        )
        sampler = ByLengthSampler(
            dataset=dst, batch_size=args.batch_size, seq_len_label="seqlen"
        )

        params = {
            "payload": dst.payload,
            "sampler": sampler,
            "device": device,
            "mode": "trai",
            "word_vocab": self.word_vocab,
            "use_simlen_pasdata": flag_use_simlen_pas_data,
            "max_pasdata_lendiff": max_pasdata_lendiff,
            "resample_target_data": resample_target_data,
            "pair_pas_with_target": pair_pas_with_target,
            "return_supervised_mask": supervised_mode,
            "pas_subsample_count": pas_subsample_count,
        }
        it = PASCtrlPCFGIter(
            langstr=langstr, prompt_type=prompt_type, return_trees=False, **params
        )
        return it


class PCFGIter(MyDataIter):
    def __init__(
        self,
        langstr,
        prompt_type,
        word_vocab,
        return_supervised_mask=False,
        return_spanoverlap_score=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.langstr = langstr
        self.word_vocab = word_vocab
        self.prompt_type = prompt_type
        self.return_supervised_mask = return_supervised_mask

    def __iter__(self):
        for batch in self.sampler:
            cleaned_form_array = [self.payload["cleaned_word_form"][i] for i in batch]
            batch_size = len(batch)
            len_array = torch.tensor([self.payload["seqlen"][i] for i in batch])
            assert max(len_array) == min(
                len_array
            ), "In batch sampling mode, all sentences should have the same length"

            cleaned_gold_trees = [self.payload["cleaned_gold_tree"][i] for i in batch]

            cleaned_id_array = torch.tensor(
                [self.word_vocab._index(i) for i in cleaned_form_array]
            )

            cleaned_gold_trees = [t for t in cleaned_gold_trees]
            assert len(cleaned_form_array) == len(
                cleaned_gold_trees
            ), "the number of sentence to test must match the number of gold trees. Esp. after pas data filtering"

            if self.return_supervised_mask:
                batch_size = len(cleaned_form_array)
                seqlen = len_array[0]
                gold_tree_mask = torch.zeros(
                    batch_size, len_array[0] + 1, len_array[0] + 1, dtype=torch.bool
                )
                for bid, tree in enumerate(cleaned_gold_trees):
                    for i in range(seqlen):
                        for j in range(i + 1, seqlen + 1):
                            if any([test_span_conflict((i, j), s[:2]) for s in tree]):
                                gold_tree_mask[bid][i, j] = False
                            else:
                                gold_tree_mask[bid][i, j] = True
                gold_tree_mask = gold_tree_mask.to(self.device)
            else:
                gold_tree_mask = None

            tree_mask = None
            w2t_mapping = None

            yield {
                "target_form_array": cleaned_form_array,
                "target_id_array": cleaned_id_array.to(self.device),
                "target_len_array": len_array,
                "w2t_mapping": w2t_mapping,
                "tree_mask": tree_mask,
            }, {"gold_tree": cleaned_gold_trees, "gold_tree_mask": gold_tree_mask}


class PCFGRewardIter(MyDataIter):
    def __init__(
        self,
        langstr,
        prompt_type,
        word_vocab,
        return_supervised_mask=False,
        return_spanoverlap_score=False,
        bert_mode=False,
        # mode_offending_spans=False,
        add_sentence_level_span=False,
        min_span_reward=-10.0,
        mode_reward=None,
        return_gold_tree=False,
        clip_info=False,
        flag_use_pos_unks = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.langstr = langstr
        self.word_vocab = word_vocab
        self.prompt_type = prompt_type
        self.return_supervised_mask = return_supervised_mask
        self.bert_mode = bert_mode
        # self.mode_offending_spans = mode_offending_spans
        self.competing_span_mask: List[Union[torch.Tensor, None]] = [
            None for _ in range(1000000)
        ]
        assert min_span_reward < 0, "min_span_reward should be negative"
        self.min_span_reward = min_span_reward
        self.add_sentence_level_span = add_sentence_level_span
        self.return_gold_tree = return_gold_tree
        print("Train Iter: add_sentence_level_span", self.add_sentence_level_span)
        # print("Train Iter: mode_offending_spans", self.mode_offending_spans)
        if self.bert_mode:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                "sentence-transformers/all-mpnet-base-v2"
            )

        assert mode_reward is not None, "mode_reward should be specified"
        self.mode_reward = mode_reward
        self.clip_info = clip_info
        self.flag_use_pos_unks = flag_use_pos_unks

    def __iter__(self):
        for batch in self.sampler:
            cleaned_form_array = [self.payload["merged_word_form"][i] for i in batch]
            batch_size = len(batch)
            len_array = torch.tensor([self.payload["merged_seqlen"][i] for i in batch])
            assert max(len_array) == min(
                len_array
            ), "In batch sampling mode, all sentences should have the same length"
            if self.return_gold_tree or self.return_supervised_mask: 
                cleaned_gold_trees = [
                    self.payload["cleaned_gold_tree"][i] for i in batch
                ]
            else:
                cleaned_gold_trees = None
            # if self.return_trees:
            # cleaned_gold_trees = [self.payload["cleaned_gold_tree"][i] for i in batch]
            # else:
            # cleaned_gold_trees = None
            if self.flag_use_pos_unks:
                pos_seq = [self.payload["cleaned_pos_array"][i] for i in batch]
                pos_seq = [[w.split('##')[0].split('-')[0] for w in s] for s in pos_seq]
            else:
                pos_seq = [None for i in batch]

            cleaned_id_array = torch.tensor(
                [self.word_vocab._index(i, p) for i, p in zip(cleaned_form_array, pos_seq)]
            )

            pas_indicator = torch.tensor(
                [self.payload["pas_indicator"][i] for i in batch]
            )

            spanoverlap_score_array = [
                self.payload["merged_spanoverlap_score"][i] for i in batch
            ]

            alignment_array = torch.zeros(
                batch_size, len_array[0] + 1, len_array[0] + 1
            )
            maxlen = len_array[0].item()
            for bid, span_group in enumerate(spanoverlap_score_array):
                if self.add_sentence_level_span:
                    alignment_array[bid, 0, maxlen] = -self.min_span_reward
                for s, c in span_group.items():
                    assert (
                        s[0] < len_array[0] and s[1] <= len_array[0] + 1
                    ), f"span {s} is out of bound with length {len_array[0], spanoverlap_score_array}"
                    # assert (
                    #     s in span_to_alignment_idx.keys()
                    # ), f"span {s} is not in the alignment index, {span_to_alignment_idx}"
                    # print(c)
                    if isinstance(c, int):
                        alignment_array[bid, s[0], s[1]] = -self.min_span_reward
                        continue
                    count_full, count_unique, ws_count_full, ws_count_unique = c[:4]
                    c_remainder = c[4:]
                    smoothing_factor = 1
                    if count_unique == 0:
                        continue

                    if self.mode_reward == "tf":
                        tf = np.log(count_unique + 1)
                        alignment_array[bid, s[0], s[1]] = max(
                            tf - self.min_span_reward, 0
                        )
                    elif self.mode_reward == "log_tfidf":
                        tf = np.log(count_unique + 1)
                        idf = np.log(c_remainder[0]/c_remainder[1])/np.log(c_remainder[0])
                        alignment_array[bid, s[0], s[1]] = max(
                            tf*idf - self.min_span_reward, 0
                        )
                    elif self.mode_reward == "log_tfidf_trivial_freq":
                        tf = np.log(count_full + 1)
                        idf = np.log(c_remainder[0]/c_remainder[1])/np.log(c_remainder[0])
                        alignment_array[bid, s[0], s[1]] = max(
                            tf*idf - self.min_span_reward, 0
                        )
                    elif self.mode_reward == "norm_tfidf":
                        tf = count_unique
                        idf = np.log(c_remainder[0]/c_remainder[1])/np.log(c_remainder[0])
                        alignment_array[bid, s[0], s[1]] = max(
                            tf*idf - self.min_span_reward, 0
                        )
                    else:
                        raise ValueError(
                            f"mode_reward {self.mode_reward} not supported"
                        )

            encoded_input = None

            def test_compatible_spans(s1, s2):
                assert isinstance(s1, torch.Tensor), f"{s1} should be tensor"
                assert isinstance(s2, list), f"{s2} should be list"
                query_shape = s1.shape
                query = s1.flatten(0, 1).unsqueeze(1)
                # query = torch.tensor(s1)[:, None, :]
                # print(s2)
                target = torch.tensor([s[:2] for s in s2])
                assert len(query.shape) == 3 and len(target.shape) == 2, f"query {query.shape} should have (b, 1, 2) and target {target.shape} should have (b, 2) shape"


                is_disjoint = np.logical_or(
                    query[:, :, 1] <= target[:, 0],  # end <= other_start
                    query[:, :, 0] >= target[:, 1]   # start >= other_end
                )
    
                is_containing = np.logical_and(
                    query[:, :, 0] <= target[:, 0],  # start <= other_start
                    query[:, :, 1] >= target[:, 1]   # end >= other_end
                )
                is_contained = np.logical_and(
                    query[:, :, 1] <= target[:, 1],  # start <= other_start
                    query[:, :, 0] >= target[:, 0]   # end >= other_end
                )

                # c1 = target[:, :, 0] < query[0] and s1[1] > s2[0] and s1[1] < s2[1]
                # c2 = s2[0] < s1[0] and s2[1] > s1[0] and s2[1] < s1[1]
                # print(is_containing, is_contained, is_disjoint)
                return (is_containing | is_contained | is_disjoint).all(dim=1).reshape(*query_shape[:2]).triu(diagonal=1)

            

            if self.return_supervised_mask:
                batch_size = len(cleaned_form_array)
                seqlen = len_array[0]
                gold_tree_mask = torch.zeros(
                    batch_size, len_array[0] + 1, len_array[0] + 1, dtype=torch.bool
                )
                span_ind_1 = torch.arange(0, seqlen+1).unsqueeze(1).expand(-1, seqlen+1)#.flatten()
                span_ind_2 = torch.arange(0, seqlen + 1).unsqueeze(0).expand(seqlen+1, -1)#.flatten()
                span_ind = torch.stack([span_ind_1, span_ind_2], dim=2)
                tree_compatible_masks = []
                for bid, tree in enumerate(cleaned_gold_trees):
                    tree_compatible_masks.append(test_compatible_spans(span_ind, tree))
                    for s in tree:
                        if s[1]-s[0] == 1:continue
                        gold_tree_mask[bid][s[0], s[1]] = True
                gold_tree_mask = gold_tree_mask.to(self.device)
                tree_compatible_masks = torch.stack(tree_compatible_masks, dim=0).to(self.device)
            else:
                gold_tree_mask = None
                tree_compatible_masks = None
            
            yield {
                "target_form_array": cleaned_form_array,
                "target_id_array": cleaned_id_array.to(self.device),
                "target_len_array": len_array,
                "reward": alignment_array.to(self.device),
                "encoded_input": encoded_input,
                "competing_span_mask": None,  # competing_span_mask.to(self.device),
                "tree_compatible_masks": tree_compatible_masks,
                "pas_indicator": pas_indicator.to(self.device),
                "gold_tree_mask": gold_tree_mask,
                # "w2t_mapping": w2t_mapping,
                # 'tree_mask': tree_mask,
            }, {"gold_tree": cleaned_gold_trees}



class DataModuleForPASCtrlPCFGReward(DataModuleForPASCtrlPCFG):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_dataloader(
        self,
        langstr,
        device="cuda",
        max_len=40,
        min_len=3,
        prompt_type="anything",
        supervised_mode=False,
        flag_curriculum_learning=False,
        pas_subsample_count=4,
        len_increment=5,
        bert_mode="disabled",
        # mode_offending_spans=False,
        add_sentence_level_span=False,
        min_span_reward=-10.0,
        mode_reward="log_tfidf",
    ):
        print("train loader: add_sentence_level_span:", add_sentence_level_span)
        print("train loader: reward mode:", mode_reward)


        def sampler_gen():
            for i in range(1, 100):
                pas_subsample_indices = [
                    random.choices(
                        [j for j in range(len(pas_group)) if len(pas_group[j]) > 5],
                        k=min(
                            pas_subsample_count, sum([len(j) > 5 for j in pas_group])
                        ),
                    )
                    for pas_group in self.train_dataset.payload["cleaned_pas_sample"]
                ]
                self.train_dataset.payload[
                    "merged_word_form"
                ] = self.train_dataset.payload["cleaned_word_form"] + list(
                    itertools.chain.from_iterable(
                        [
                            [g[idx] for idx in ids]
                            for g, ids in zip(
                                self.train_dataset.payload["cleaned_pas_sample"],
                                pas_subsample_indices,
                            )
                        ]
                    )
                )
                self.train_dataset.payload[
                    "merged_spanoverlap_score"
                ] = self.train_dataset.payload["trg_spanoverlap_score"] + list(
                    itertools.chain.from_iterable(
                        [
                            [g[idx] for idx in ids]
                            for g, ids in zip(
                                self.train_dataset.payload["pas_spanoverlap_score"],
                                pas_subsample_indices,
                            )
                        ]
                    )
                )
                self.train_dataset.payload["pas_indicator"] = [0] * len(
                    self.train_dataset.payload["cleaned_word_form"]
                ) + [1] * len(
                    list(
                        itertools.chain.from_iterable(
                            [
                                [g[idx] for idx in ids]
                                for g, ids in zip(
                                    self.train_dataset.payload["cleaned_pas_sample"],
                                    pas_subsample_indices,
                                )
                            ]
                        )
                    )
                )
                self.train_dataset.payload["merged_seqlen"] = [
                    len(i) for i in self.train_dataset.payload["merged_word_form"]
                ]
                args = self.hparams.train
                dst = self.train_dataset.drop(
                    lambda x: x > max_len or x < min_len, payload_label="merged_seqlen"
                )
                print("finished pruning dataset")
                sampler = ByLengthSamplerV1(
                    dataset=dst,
                    batch_size=args.batch_size,
                    seq_len_label="merged_seqlen",
                    train=True,
                    # curriculum_learning=flag_curriculum_learning,
                    max_len=(
                        min(max_len, 20 + i * len_increment)
                        if flag_curriculum_learning
                        else max_len
                    ),
                    max_epoch=1,
                )

                params = {
                    "payload": dst.payload,
                    "sampler": sampler,
                    "device": device,
                    "mode": "trai",
                    "word_vocab": self.word_vocab,
                    "bert_mode": bert_mode,
                    "return_supervised_mask": supervised_mode,
                    # "mode_offending_spans": mode_offending_spans,
                    "add_sentence_level_span": add_sentence_level_span,
                    "min_span_reward": min_span_reward,
                    "mode_reward": mode_reward,
                    "return_gold_tree": False,
                    "flag_use_pos_unks": self.flag_use_pos_unks
                }
                it = PCFGRewardIter(
                    langstr=langstr,
                    prompt_type=prompt_type,
                    #  add_sentence_level_span=add_sentence_level_span,
                    **params,
                )
                yield from it

        return sampler_gen(), 1000000

    def dev_dataloader(
        self,
        langstr,
        device="cuda",
        prompt_type="anything",
        max_len=40,
        min_len=3,
    ):
        args = self.hparams.test
        val_dataset = self.val_dataset.drop(
            lambda x: x > max_len or x < min_len, payload_label="seqlen"
        )
        val_sampler = ByLengthSamplerV1(
            dataset=val_dataset,
            batch_size=args.batch_size,
            seq_len_label="seqlen",
            max_len=max_len,
        )

        params = {
            "payload": val_dataset.payload,
            "sampler": val_sampler,
            "device": device,
            "mode": "eval",
            "word_vocab": self.word_vocab,
        }
        it = PCFGIter(langstr=langstr, prompt_type=prompt_type, **params)
        return it, val_sampler.size

    def dev_full_dataloader(
        self,
        langstr,
        device="cuda",
        prompt_type="anything",
        max_len=40,
        min_len=3,
        bert_mode=False,
        # mode_offending_spans=True,
        add_sentence_level_span=False,
        min_span_reward=-10.0,
        mode_reward="semantic information",
    ):
        print("dev full loader: add_sentence_level_span:", add_sentence_level_span)
        print("train loader: reward mode:", mode_reward)
        args = self.hparams.test
        pas_subsample_indices = [
            random.choices(range(len(pas_group)), k=min(0, len(pas_group)))
            for pas_group in self.val_dataset.payload["cleaned_pas_sample"]
        ]
        self.val_dataset.payload["merged_word_form"] = self.val_dataset.payload[
            "cleaned_word_form"
        ] + list(
            itertools.chain.from_iterable(
                [
                    [g[idx] for idx in ids]
                    for g, ids in zip(
                        self.val_dataset.payload["cleaned_pas_sample"],
                        pas_subsample_indices,
                    )
                ]
            )
        )
        self.val_dataset.payload["merged_spanoverlap_score"] = self.val_dataset.payload[
            "trg_spanoverlap_score"
        ] + list(
            itertools.chain.from_iterable(
                [
                    [g[idx] for idx in ids]
                    for g, ids in zip(
                        self.val_dataset.payload["pas_spanoverlap_score"],
                        pas_subsample_indices,
                    )
                ]
            )
        )
        self.val_dataset.payload["pas_indicator"] = [0] * len(
            self.val_dataset.payload["cleaned_word_form"]
        ) + [1] * len(
            list(
                itertools.chain.from_iterable(
                    [
                        [g[idx] for idx in ids]
                        for g, ids in zip(
                            self.val_dataset.payload["cleaned_pas_sample"],
                            pas_subsample_indices,
                        )
                    ]
                )
            )
        )
        self.val_dataset.payload["merged_seqlen"] = [
            len(i) for i in self.val_dataset.payload["merged_word_form"]
        ]
        args = self.hparams.train
        dst = self.val_dataset.drop(
            lambda x: x > max_len or x < min_len, payload_label="merged_seqlen"
        )
        print("finished pruning dataset, current dataset length", len(dst.payload["merged_seqlen"]))
        val_sampler = ByLengthSamplerV1(
            dataset=dst,
            batch_size=args.batch_size,
            seq_len_label="merged_seqlen",
            # curriculum_learning=flag_curriculum_learning,
            max_len=max_len,  # min(max_len, 20 + i*len_increment) if flag_curriculum_learning else max_len,
            max_epoch=1,
        )

        params = {
            "payload": dst.payload,
            "sampler": val_sampler,
            "device": device,
            "mode": "trai",
            "word_vocab": self.word_vocab,
            "bert_mode": bert_mode,
            "add_sentence_level_span": add_sentence_level_span,
            "min_span_reward": min_span_reward,
            "mode_reward": mode_reward,
            "return_gold_tree": True,
            "flag_use_pos_unks": self.flag_use_pos_unks
        }
        it = PCFGRewardIter(langstr=langstr, prompt_type=prompt_type, **params)

        return it, val_sampler.size

    def test_full_dataloader(
        self,
        langstr,
        device="cuda",
        prompt_type="anything",
        max_len=40,
        min_len=3,
        bert_mode=False,
        # mode_offending_spans=True,
        add_sentence_level_span=False,
        min_span_reward=-10.0,
        mode_reward="semantic information",
    ):
        print("dev full loader: add_sentence_level_span:", add_sentence_level_span)
        print("train loader: reward mode:", mode_reward)
        args = self.hparams.test
        pas_subsample_indices = [
            random.choices(range(len(pas_group)), k=min(0, len(pas_group)))
            for pas_group in self.test_dataset.payload["cleaned_pas_sample"]
        ]
        self.test_dataset.payload["merged_word_form"] = self.test_dataset.payload[
            "cleaned_word_form"
        ] + list(
            itertools.chain.from_iterable(
                [
                    [g[idx] for idx in ids]
                    for g, ids in zip(
                        self.test_dataset.payload["cleaned_pas_sample"],
                        pas_subsample_indices,
                    )
                ]
            )
        )
        self.test_dataset.payload["merged_spanoverlap_score"] = self.test_dataset.payload[
            "trg_spanoverlap_score"
        ] + list(
            itertools.chain.from_iterable(
                [
                    [g[idx] for idx in ids]
                    for g, ids in zip(
                        self.test_dataset.payload["pas_spanoverlap_score"],
                        pas_subsample_indices,
                    )
                ]
            )
        )
        self.test_dataset.payload["pas_indicator"] = [0] * len(
            self.test_dataset.payload["cleaned_word_form"]
        ) + [1] * len(
            list(
                itertools.chain.from_iterable(
                    [
                        [g[idx] for idx in ids]
                        for g, ids in zip(
                            self.test_dataset.payload["cleaned_pas_sample"],
                            pas_subsample_indices,
                        )
                    ]
                )
            )
        )
        self.test_dataset.payload["merged_seqlen"] = [
            len(i) for i in self.test_dataset.payload["merged_word_form"]
        ]
        args = self.hparams.train
        dst = self.test_dataset.drop(
            lambda x: x > max_len or x < min_len, payload_label="merged_seqlen"
        )
        print("finished pruning dataset, current dataset length", len(dst.payload["merged_seqlen"]))
        val_sampler = ByLengthSamplerV1(
            dataset=dst,
            batch_size=args.batch_size,
            seq_len_label="merged_seqlen",
            # curriculum_learning=flag_curriculum_learning,
            max_len=max_len,  # min(max_len, 20 + i*len_increment) if flag_curriculum_learning else max_len,
            max_epoch=1,
        )

        params = {
            "payload": dst.payload,
            "sampler": val_sampler,
            "device": device,
            "mode": "trai",
            "word_vocab": self.word_vocab,
            "bert_mode": bert_mode,
            # "mode_offending_spans": mode_offending_spans,
            "add_sentence_level_span": add_sentence_level_span,
            "min_span_reward": min_span_reward,
            "mode_reward": mode_reward,
            "return_gold_tree": True,
            "flag_use_pos_unks": self.flag_use_pos_unks
        }
        it = PCFGRewardIter(langstr=langstr, prompt_type=prompt_type, **params)

        return it, val_sampler.size

    def test_dataloader(
        self,
        langstr,
        device="cuda",
        prompt_type="anything",
        max_len=40,
        min_len=3,
    ):
        args = self.hparams.test
        val_dataset = self.test_dataset.drop(
            lambda x: x > max_len or x < min_len, payload_label="seqlen"
        )
        val_sampler = ByLengthSamplerV1(
            dataset=val_dataset,
            batch_size=args.batch_size,
            seq_len_label="seqlen",
            max_len=max_len,
        )

        params = {
            "payload": val_dataset.payload,
            "sampler": val_sampler,
            "device": device,
            "mode": "eval",
            "word_vocab": self.word_vocab,
        }
        it = PCFGIter(langstr=langstr, prompt_type=prompt_type, **params)
        return it, val_sampler.size


