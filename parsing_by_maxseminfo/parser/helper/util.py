import time
import os
import logging
from distutils.dir_util import copy_tree

from parsing_by_maxseminfo.parser.model import NeuralPCFG, CompoundPCFG, TNPCFG, FastTNPCFG, Simple_N_PCFG, Simple_C_PCFG

import torch
import numpy as np


def get_model(args, dataset):
    if args.model_name == 'NPCFG':
        return NeuralPCFG(args, dataset).to(dataset.device)

    elif args.model_name == 'CPCFG':
        return CompoundPCFG(args, dataset).to(dataset.device)

    elif args.model_name == 'TNPCFG':
        return TNPCFG(args, dataset).to(dataset.device)


    elif args.model_name == 'NLPCFG':
        return NeuralLPCFG(args, dataset).to(dataset.device)

    elif args.model_name == 'NBLPCFG':
        return NeuralBLPCFG(args, dataset).to(dataset.device)

    elif args.model_name == 'FastTNPCFG':
        return FastTNPCFG(args, dataset).to(dataset.device)

    elif args.model_name == 'FastNBLPCFG':
        return FastNBLPCFG(args, dataset).to(dataset.device)
    
    elif args.model_name == "SNPCFG":
        return Simple_N_PCFG(args, dataset).to(dataset.device)
    
    elif args.model_name == "SCPCFG":
        return Simple_C_PCFG(args, dataset).to(dataset.device)

    else:
        raise KeyError


def get_optimizer(args, model):
    if args.name == 'adam':
        return torch.optim.Adam(params=model.parameters(), lr=args.lr, betas=(args.mu, args.nu))
    elif args.name == 'adamw':
        return torch.optim.AdamW(params=model.parameters(), lr=args.lr, betas=(args.mu, args.nu), weight_decay=args.weight_decay)
    else:
        raise NotImplementedError

def get_logger(args, log_name='train',path=None):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    handler = logging.FileHandler(os.path.join(args.save_dir if path is None else path, '{}.log'.format(log_name)), 'w')
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)
    logger.propagate = False
    logger.info(args)
    return logger


def create_save_path(args):
    model_name = args.model.model_name
    suffix = "/{}".format(model_name) + time.strftime("%Y-%m-%d-%H_%M_%S",
                                                                             time.localtime(time.time()))
    from pathlib import Path
    saved_name = Path(args.save_dir).stem + suffix
    args.save_dir = args.save_dir + suffix

    if os.path.exists(args.save_dir):
        print(f'Warning: the folder {args.save_dir} exists.')
    else:
        print('Creating {}'.format(args.save_dir))
        os.makedirs(args.save_dir)
    # save the config file and model file.
    import shutil
    shutil.copyfile(args.conf, args.save_dir + "/config.yaml")
    os.makedirs(args.save_dir + "/parser")
    copy_tree("parser/", args.save_dir + "/parser")
    return  saved_name

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

    def score_by_longest_matches(
        self,
        form,
        sample,
        stemmer,
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
        assert isinstance(form, list), f"{form} must be pre-tokenized"

        if spanoverlap_mask is None:
            spanoverlap_mask = np.ones((len(form)+1, len(form)+1), dtype=bool)


        cleaned_sample = [stemmer.stemWord(self.convert_to_ascii(w)) for w in sample]
        cleaned_form = [stemmer.stemWord(self.convert_to_ascii(w)) for w in form]
        len_form = len(cleaned_form)
        len_sample = len(cleaned_sample)

        hits = []


        scores = np.zeros((len_form + 1, len_form+1, len_sample+1, len_sample + 1), dtype=float)
        for w in range(len(form), 1, -1):
            for i in range(len(form) - w + 1):
                hit_check = [hit[0] <= i and hit[1] >= i + w for hit in hits]
                if any(hit_check):
                    # print("skip due to being contained in a longer match")
                    continue
                for j in range(len(sample) - w + 1):
                    if test_equality(cleaned_form[i:i+w], cleaned_sample[j:j+w]):
                        if spanoverlap_mask[i][i+w]:
                            scores[i, i+w, j, j+w] = 1
                            hits.append((i, i+w))
                        else:
                            continue
        return scores

