# %%
"""

"""

# %%
from easydict import EasyDict as edict
import torch

# %%
import numpy as np
import random
import torch

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# %%


# %%
from parsing_by_maxseminfo.utils.myargparse import get_argsndevice
# utils.myargparse import get_argsndevice
args, device = get_argsndevice()

# %%

print("processing data")
print(args.data)

# %%
args.train.batch_size = 4
args.test.batch_size = 8

print("processing use spacy: ", args.preprocessing_use_spacy)

# %%
from parsing_by_maxseminfo.parser.helper.pas_grammar_data_helper import DataModuleForPASCtrlPCFG
dst = DataModuleForPASCtrlPCFG(args, langstr=args.data.language, use_cache=False, max_size=args.vocab_max_size, merge_pas_data=False, flag_use_spacy_preprocessing=args.preprocessing_use_spacy, flag_spanoverlap_match_char=False, flag_use_spacy_for_treebank=args.preprocessing_use_spacy, pas_subsample=args.preprocessing_pas_subsample_count, flag_compute_relative_frequency=args.flag_compute_relative_frequency, distinct_prompt=args.distinct_prompt)
