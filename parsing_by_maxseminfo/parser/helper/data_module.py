import pickle
# from parsing_by_maxseminfo.fastNLP.core.dataset import DataSet
# from parsing_by_maxseminfo.fastNLP.core.batch import DataSetIter
# from parsing_by_maxseminfo.fastNLP.core.vocabulary import Vocabulary
# from parsing_by_maxseminfo.fastNLP.embeddings import StaticEmbedding
# from parsing_by_maxseminfo.fastNLP.core.sampler import BucketSampler, ConstantTokenNumSampler
from torch.utils.data import Sampler
from collections import defaultdict
import os
import random


class DataModule:
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.device = self.hparams.device
        self.setup()

    def prepare_data(self):
        pass

    def setup(self):
        raise NotImplementedError("Deprecated")
    
    def train_dataloader(self, max_len=40):
        raise NotImplementedError("Deprecated")

    @property
    def val_dataloader(self):
        raise NotImplementedError("Deprecated")

    
    def test_dataloader(self):
        raise NotImplementedError("Deprecated")

"""
Same as (Kim et al, 2019)
"""


class ByLengthSampler(Sampler):
    def __init__(self, dataset, batch_size=4, seq_len_label="seqlen"):
        self.group = defaultdict(list)
        if "payload" in dataset.__dict__.keys():
            self.seq_lens = dataset.payload[seq_len_label]
        else:
            self.seq_lens = dataset[seq_len_label]
        # self.seq_lens = dataset['seq_len']
        for i, length in enumerate(self.seq_lens):
            self.group[length].append(i)
        self.batch_size = batch_size
        total = []

        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i : i + n]

        for idx, lst in self.group.items():
            total = total + list(chunks(lst, self.batch_size))
        self.total = total

    def __iter__(self):
        random.shuffle(self.total)
        for batch in self.total:
            yield batch

    def __len__(self):
        return len(self.total)


class ByLengthSamplerV1(Sampler):
    def __init__(
        self,
        dataset,
        batch_size=4,
        seq_len_label="seqlen",
        train=False,
        curriculum_learning=False,
        min_len=15,
        max_len=100,
        len_increment=5,
        max_epoch=25,
    ):
        self.group = defaultdict(list)
        if "payload" in dataset.__dict__.keys():
            self.seq_lens = dataset.payload[seq_len_label]
        else:
            self.seq_lens = dataset[seq_len_label]
        # self.seq_lens = dataset['seq_len']
        for i, length in enumerate(self.seq_lens):
            self.group[length].append(i)
        self.batch_size = batch_size
        if train:
            from tqdm import tqdm

            print("Constructing shuffled batches with 50 epoches")
            if curriculum_learning:
                self.total = sum(
                    [
                        self._yield_shuffled_data_source(
                            maxlen=min(max_len, min_len + _ * len_increment),
                            shuffle=True,
                        )
                        for _ in tqdm(range(max_epoch))
                    ],
                    [],
                )
            else:
                self.total = sum(
                    [
                        self._yield_shuffled_data_source(shuffle=True, maxlen=max_len)
                        for _ in tqdm(range(max_epoch))
                    ],
                    [],
                )
        else:
            print(
                "sampling: current dataset size: ",
                sum([len(v) for k, v in self.group.items()]),
            )
            self.total = sum(
                [
                    self._yield_shuffled_data_source(shuffle=False, maxlen=max_len)
                    for _ in range(1)
                ],
                [],
            )

    def _yield_shuffled_data_source(self, maxlen, shuffle=False):
        # print("filling data")
        total = []

        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                if shuffle:
                    random.shuffle(lst)
                yield lst[i : i + n]

        for idx, lst in self.group.items():
            if idx > maxlen:
                continue
            total = total + list(chunks(lst, self.batch_size))
        if shuffle:
            random.shuffle(total)
        self.size = len(total) + 10
        return total

    def __iter__(self):
        # total = sum([self._yield_shuffled_data_source() for _ in range(1000)])
        for batch in self.total:
            yield batch

    def __len__(self):
        return len(self.total)
