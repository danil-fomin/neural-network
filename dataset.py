import json
from pathlib import Path

from torch.utils.data import Dataset


class ReviewsDataset(Dataset):
    def __init__(self, jsonl_path):
        self.jsonl_path = Path(jsonl_path)
        self.samples = []

        with open(self.jsonl_path, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                self.samples.append((record["text"], record["label"]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        text, label = self.samples[index]
        return (text, label)


class BowReviewsDataset(Dataset):
    def __init__(self, base_dataset, vocabulary):
        self.base_dataset = base_dataset
        self.vocabulary = vocabulary

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        text, label = self.base_dataset[idx]
        bow = self.vocabulary.text_to_bow(text)
        return (bow, label)
