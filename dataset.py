from typing import List, Dict
import torch

from torch.utils.data import Dataset

from utils import Vocab, pad_to_len
import re


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn

        results = {"text": [], "intent": [], "id": []}
        for data in samples:
            results["id"].append(data["id"])
            results["text"].append(re.sub(r"(\w)([^a-zA-Z0-9 ])", r"\1 \2", data["text"]).split(" "))
            if "intent" in data:
                results["intent"].append(self.label_mapping[data["intent"]])
        else:
            results["text"] = torch.LongTensor(self.vocab.encode_batch(results["text"], self.max_len))
            results["intent"] = torch.LongTensor(results["intent"])

        # raise NotImplementedError
        return results

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SeqTaggingClsDataset(SeqClsDataset):
    ignore_idx = -100

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        results = {"tokens": [], "tags": [], "id": []}
        for data in samples:
            results["id"].append(data["id"])
            results["tokens"].append(data["tokens"])
            if "tags" in data:
                results["tags"].append(list(map(lambda x: self.label_mapping[x], data["tags"])))
        else:
            results["tokens"] = torch.LongTensor(self.vocab.encode_batch(results["tokens"], self.max_len))
            results["tags"] = torch.LongTensor(pad_to_len(results["tags"], self.max_len, 9))

        # raise NotImplementedError
        return results
