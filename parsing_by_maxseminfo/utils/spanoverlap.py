from matplotlib.pyplot import flag
from networkx import hits, jaccard_coefficient
import numpy as np
from nltk import word_tokenize
import random

class SpanScorer:
    def __init__(self) -> None:
        # self.form = form
        pass

    def checkbow(self, a, b):
        from collections import Counter
        ac = Counter(a)
        bc = Counter(b)
        return ac == bc
    
    def checksubstring(self, a, b):
        return a in b


    def score_by_count(
        self,
        form,
        sample,
        stemmer,
        flag_rm_spaces_in_matching=False,
        flag_print_samples=False,
        flag_shuffle_samples = False
    ):
        from ipynb.utils import normalizing_string
        assert isinstance(form, list), f"{form} must be pre-tokenized"

        #! samples should be lists of pre-tokenized words
        #! forms should also be pre-tokenized

        cleaned_sample = [stemmer.stemWord(w) for w in sample]
        cleaned_form = [stemmer.stemWord(w) for w in form]
        len_form = len(cleaned_form)
        len_sample = len(cleaned_sample)

        scores = np.zeros((len_form + 1, len_form+1, len_sample+1, len_sample + 1), dtype=float)
        for w in range(2, len(form)):
            for i in range(len(form) - w + 1):
                for j in range(len(sample) - w + 1):
                    if cleaned_form[i:i+w] == cleaned_sample[j:j+w]:
                        scores[i, i+w, j, j+w] = 1

        return scores
    
    def score_by_weighted_longest_matches(
        self,
        form,
        sample,
        stemmer,
        flag_rm_spaces_in_matching=False,
        flag_print_samples=False,
        flag_shuffle_samples = False
    ):
        from ipynb.utils import normalizing_string
        assert isinstance(form, list), f"{form} must be pre-tokenized"

        #! samples should be lists of pre-tokenized words
        #! forms should also be pre-tokenized

        cleaned_sample = [stemmer.stemWord(w) for w in sample]
        cleaned_form = [stemmer.stemWord(w) for w in form]
        len_form = len(cleaned_form)
        len_sample = len(cleaned_sample)

        hits = []
        
        weight_constant = [1] * 100
        weight = [0] + list(range(100))
        weight_exp = np.exp(weight)
        # fib = lambda n: [a := 0, b := 1] + [a := b, b := a + b][1] * (n - 2)

        scores = np.zeros((len_form + 1, len_form+1, len_sample+1, len_sample + 1), dtype=float)
        for w in range(len(form)-1, 1, -1):
            for i in range(len(form) - w + 1):
                hit_check = [hit[0] <= i and hit[1] >= i + w for hit in hits]
                if any(hit_check):
                    # print("skip due to being contained in a longer match")
                    continue
                for j in range(len(sample) - w + 1):
                    if cleaned_form[i:i+w] == cleaned_sample[j:j+w]:
                        scores[i, i+w, j, j+w] = weight[w]
                        hits.append((i, i+w))
                        # print(hits)
        return scores


    def convert_to_ascii(self, text):
        text = text.lower()
        replacements = {
            'Ä': 'Ae', 'ä': 'ae', 'Ö': 'Oe', 'ö': 'oe', 'Ü': 'Ue', 'ü': 'ue', 'ß': 'ss',
            'À': 'A', 'Á': 'A', 'Â': 'A', 'Ã': 'A', 'Å': 'A', 'Æ': 'Ae', 'Ç': 'C', 'È': 'E', 'É': 'E', 'Ê': 'E', 'Ë': 'E',
            'Ì': 'I', 'Í': 'I', 'Î': 'I', 'Ï': 'I', 'Ð': 'D', 'Ñ': 'N', 'Ò': 'O', 'Ó': 'O', 'Ô': 'O', 'Õ': 'O', 'Ø': 'O',
            'Ù': 'U', 'Ú': 'U', 'Û': 'U', 'Ý': 'Y', 'Þ': 'Th', 'à': 'a', 'á': 'a', 'â': 'a', 'ã': 'a', 'å': 'a', 'æ': 'ae',
            'ç': 'c', 'è': 'e', 'é': 'e', 'ê': 'e', 'ë': 'e', 'ì': 'i', 'í': 'i', 'î': 'i', 'ï': 'i', 'ð': 'd', 'ñ': 'n',
            'ò': 'o', 'ó': 'o', 'ô': 'o', 'õ': 'o', 'ø': 'o', 'ù': 'u', 'ú': 'u', 'û': 'u', 'ý': 'y', 'þ': 'th', 'ÿ': 'y'
        }
        
        for char, ascii_equiv in replacements.items():
            text = text.replace(char, ascii_equiv)
        
        return text

    def get_ngram_list(self, textlist, n):
        ngrams = []
        for i in range(len(textlist) - n + 1):
            ngrams.append(" ".join(textlist[i:i+n]))
        return ngrams

    def jaccard_similarity(self, a, b):
        a = set(self.get_ngram_list(a, 1) + self.get_ngram_list(a, 2)+ self.get_ngram_list(a, 3))
        b = set(self.get_ngram_list(b, 1) + self.get_ngram_list(b, 2)+ self.get_ngram_list(b, 3))
        # b = set(b)
        return len(a.intersection(b)) / len(a.union(b))

    def score_by_jaccard_similarity(
        self,
        form,
        sample,
        stemmer,
        flag_rm_spaces_in_matching=False,
        flag_print_samples=False,
        flag_shuffle_samples = False,
        rb_bias = False,
        flag_print = False
    ):
        from ipynb.utils import normalizing_string
        assert isinstance(form, list), f"{form} must be pre-tokenized"

        #! samples should be lists of pre-tokenized words
        #! forms should also be pre-tokenized

        cleaned_sample = [stemmer.stemWord(self.convert_to_ascii(w)) for w in sample]
        cleaned_form = [stemmer.stemWord(self.convert_to_ascii(w)) for w in form]
        len_form = len(cleaned_form)
        len_sample = len(cleaned_sample)

        hits = []
        if flag_print:
            print("====")
            print(" ".join(cleaned_form))
            print(" ".join(cleaned_sample))
            print()

        scores = np.zeros((len_form + 1, len_form+1, len_sample+1, len_sample + 1), dtype=float)
        for w in range(len(form), 1, -1):
            for i in range(len(form) - w + 1):
                hit_check = [hit[0] <= i and hit[1] >= i + w for hit in hits]
                if any(hit_check):
                    # print("skip due to being contained in a longer match")
                    continue
                for j in range(len(sample) - w + 1):
                    # if flag_print:
                        # print(cleaned_form[i:i+w], cleaned_sample[j:j+w])
                    scores[i, i+w, j, j+w] = self.jaccard_similarity(cleaned_form[i:i+w], cleaned_sample[j:j+w])
                    if False and self.jaccard_similarity(cleaned_form[i:i+w], cleaned_sample[j:j+w]) >0.8:
                    # if cleaned_form[i:i+w] == cleaned_sample[j:j+w]:
                        scores[i, i+w, j, j+w] = self.jaccard_similarity(cleaned_form[i:i+w], cleaned_sample[j:j+w])
                        if rb_bias:
                            for k in range(i, i+w-1):
                                scores[k, i+w, 1, 0] = 0.1
                        hits.append((i, i+w))
                        # print(hits)
        return scores


    def score_by_longest_matches_onesided(
        self,
        form,
        sample,
        stemmer,
        flag_rm_spaces_in_matching=False,
        flag_print_samples=False,
        flag_shuffle_samples = False,
        rb_bias = False,
        flag_print = False,
        random_alignment_p = 0.,
        spanoverlap_mask = None,
        match_character_only = False,):

        def remove_punct(x):
            import string
            return x.translate(str.maketrans('', '', string.punctuation))
        
        cleaned_sample = [stemmer.stemWord(self.convert_to_ascii(w)) for w in sample]
        cleaned_form = [stemmer.stemWord(self.convert_to_ascii(w)) for w in form]
        len_form = len(cleaned_form)
        len_sample = len(cleaned_sample)
        scores = np.zeros((len_form + 1, len_form+1, len_sample+1, len_sample + 1), dtype=float)


        hits = []
        reference_string = remove_punct(" ".join(cleaned_sample))
        
        for w in range(len(form), 1, -1):
            for i in range(len(form) - w + 1):
                hit_check = [hit[0] <= i and hit[1] >= i + w for hit in hits]
                if any(hit_check):
                    # print("skip due to being contained in a longer match")
                    continue
                if remove_punct(" ".join(cleaned_form[i:i+w])) in reference_string:
                    scores[i, i+w, 0, 2] = 1
                    hits.append((i, i+w))
        return scores


    def score_by_longest_matches(
        self,
        form,
        sample,
        stemmer,
        flag_rm_spaces_in_matching=False,
        flag_print_samples=False,
        flag_shuffle_samples = False,
        rb_bias = False,
        flag_print = False,
        random_alignment_p = 0.,
        spanoverlap_mask = None,
        match_character_only = False,
    ):
        
        def remove_punct(x):
            import string
            return x.translate(str.maketrans('', '', string.punctuation))
        def test_equality(a, b):
            if match_character_only and len(a) > 2:
                return  remove_punct(''.join(a)) == remove_punct(''.join(b))
            else:
                return a == b
        from ipynb.utils import normalizing_string
        assert isinstance(form, list), f"{form} must be pre-tokenized"

        if spanoverlap_mask is None:
            spanoverlap_mask = np.ones((len(form)+1, len(form)+1), dtype=bool)

        #! samples should be lists of pre-tokenized words
        #! forms should also be pre-tokenized

        cleaned_sample = [stemmer.stemWord(self.convert_to_ascii(w)) for w in sample]
        cleaned_form = [stemmer.stemWord(self.convert_to_ascii(w)) for w in form]
        len_form = len(cleaned_form)
        len_sample = len(cleaned_sample)

        hits = []

        if flag_print:
            print("====")
            print(" ".join(cleaned_form))
            print(" ".join(cleaned_sample))
            print()

        scores = np.zeros((len_form + 1, len_form+1, len_sample+1, len_sample + 1), dtype=float)
        for w in range(len(form), 1, -1):
            for i in range(len(form) - w + 1):
                hit_check = [hit[0] <= i and hit[1] >= i + w for hit in hits]
                if any(hit_check):
                    # print("skip due to being contained in a longer match")
                    continue
                for j in range(len(sample) - w + 1):
                    # if flag_print:
                        # print(cleaned_form[i:i+w], cleaned_sample[j:j+w])
                    if random_alignment_p > 0.1:
                        if random.random() < random_alignment_p:
                            scores[i, i+w, j, j+w] = 1
                            hits.append((i, i+w))
                    else:
                        if test_equality(cleaned_form[i:i+w], cleaned_sample[j:j+w]):
                            # if rb_bias:
                                # for k in range(i, i+w-1):
                                    # scores[k, i+w, 1, 0] = 0.1
                            if spanoverlap_mask[i][i+w]:
                                scores[i, i+w, j, j+w] = 1
                                hits.append((i, i+w))
                            else:
                                continue
                        # print(hits)
        return scores

    def score_by_bow(
        self,
        form,
        sample,
        stemmer,
        flag_rm_spaces_in_matching=False,
        flag_print_samples=False,
        flag_shuffle_samples = False
    ):
        from ipynb.utils import normalizing_string
        from collections import Counter
        assert isinstance(form, list), f"{form} must be pre-tokenized"


        #! samples should be lists of pre-tokenized words
        #! forms should also be pre-tokenized

        cleaned_sample = [stemmer.stemWord(w) for w in sample]
        cleaned_form = [stemmer.stemWord(w) for w in form]
        len_form = len(cleaned_form)
        len_sample = len(cleaned_sample)

        scores = np.zeros((len_form + 1, len_form+1, len_sample+1, len_sample + 1), dtype=bool)
        for w in range(2, len(form)):
            for i in range(len(form) - w + 1):
                c1 = Counter(cleaned_form[i:i+w])
                for j in range(len(sample) - w + 1):
                    if c1 == Counter(cleaned_sample[j:j+w]):
                        scores[i, i+w, j, j+w] = 1
        return scores



