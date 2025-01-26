from math import e
import torch
import string
from nltk import PorterStemmer
from nltk import word_tokenize
import numpy as np

ps = PorterStemmer()


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


nlp = [None, None]
import spacy


def normalizing_string_by_lemma(strg, lang_str):
    strg = strg.translate(str.maketrans("", "", string.punctuation)).lower()
    if lang_str is None:
        return normalizing_identity(strg)
    if nlp[0] is None or nlp[1] != lang_str:
        nlp[0] = spacy.load(
            lang_str,
            disable=["parser", "senter", "ner"],
        )
        nlp[1] = lang_str
    processed = nlp[0](strg)
    strg = " ".join([w.lemma_.strip() for w in processed if len(w.lemma_.strip()) > 0])
    return strg


def normalizing_identity(strg):
    return strg


class SpacyLemmatizer:

    def __init__(self, langstr) -> None:
        import spacy

        langstr2spacy_module = {"korean": "ko_core_news_sm"}
        self.lemmatizer = spacy.load(langstr2spacy_module[langstr])
        pass

    def lemmatize(self, text):
        doc = self.lemmatizer(text)
        lemmas = []

        # print(doc.text)

        for token in doc:
            # print(token.lemma_.split('+'))#, token.morph.__str__())
            lemmas.extend(token.lemma_.split("+"))
        return "".join(lemmas)


def normalizing_string(strg, stemmer):
    # return normalizing_string_by_lemma_he(strg, stemmer)
    # print(stemmer, isinstance(stemmer, SL), type(stemmer))
    return normalizing_string_by_stemmer(strg, stemmer)


def normalizing_string_by_lemma(strg, stemmer):
    strg = strg.translate(str.maketrans("", "", string.punctuation)).lower()
    strg = stemmer.lemmatize(strg)
    return strg


def removing_punctuations(strg):

    mypunct = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    strg = strg.translate(str.maketrans("", "", mypunct)).lower()
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
