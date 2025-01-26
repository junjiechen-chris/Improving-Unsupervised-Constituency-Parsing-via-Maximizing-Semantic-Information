from torch.utils.data import Dataset
import itertools

class MyDataIter(Dataset):
    def __init__(
        self,
        payload,
        sampler,
        # model=None,
        # tokenizer=None,
        word_vocab=None,
        other_params={},
        device="cuda:0",
        mode="train",
        # pos_vocab=None,
        # mask_prob=0.15,
    ) -> None:
        super().__init__()
        self.payload = payload
        self.sampler = sampler
        self.device = device
        self.other_params = other_params
        self.word_vocab = word_vocab
        self.mode = mode

    def __iter__(self):
        raise NotImplementedError("HFDatasetIter is a base class")

    def __len__(self):
        return len(self.sampler)
        # for batch in self.sampler:
        #     data = [self.data.input_ids[i] for i in batch]
        #     yield huggingfaceDataCollator(data, self.model, device = self.device), None

class MyDataset(Dataset):
    def __init__(self, **kwargs):
        self.payload = {**kwargs}

    @classmethod
    def from_pickle(cls, pickle_file):
        payload = {**pickle_file}
        return cls(**payload)

    def add_field(self, store_label, fn, source_label, flag_concat=False):
        # with Pool(2) as p:
        self.payload[store_label] = list(
            map(fn, self.payload[source_label])
        )  # [fn(i) for i in self.payload[source_label]]
        if flag_concat:
            print("cating the payloads")
            self.payload[store_label] = list(
                itertools.chain.from_iterable(self.payload[store_label])
            )  # sum(self.payload[store_label], [])

    def apply_field(self, fn, label):
        for i in range(len(self.payload[label])):
            self.payload[label][i] = fn(self.payload[label][i])

    def drop(self, lbd, payload_label):
        data = self.payload[payload_label]
        mask = [lbd(x) for x in data]
        # print(self.payload['logprobs'])
        new_payload = {
            k: [x for x, m in zip(v, mask) if not m] for k, v in self.payload.items()
        }
        # print(new_payload['logprobs'])
        return MyDataset(**new_payload)

    def __getitem__(self, key):
        return self.__dict__[key]
