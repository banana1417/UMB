import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class UMBFakeNewsDataset(Dataset):
    """Dataset for title, image, true rationale, false rationale, and label."""

    def __init__(self, records, tokenizer, image_processor, max_length=128):
        self.records = list(records)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def _tokenize(self, text):
        return self.tokenizer(
            str(text or ""),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record["image_path"]) as image:
            pixel_values = self.image_processor(
                images=image.convert("RGB"), return_tensors="pt"
            )["pixel_values"][0]

        title = self._tokenize(record.get("title", ""))
        true_reason = self._tokenize(record.get("true_reason", ""))
        false_reason = self._tokenize(record.get("false_reason", ""))

        return {
            "image": pixel_values,
            "title_input_ids": title["input_ids"][0],
            "title_attention_mask": title["attention_mask"][0],
            "true_reason_input_ids": true_reason["input_ids"][0],
            "true_reason_attention_mask": true_reason["attention_mask"][0],
            "false_reason_input_ids": false_reason["input_ids"][0],
            "false_reason_attention_mask": false_reason["attention_mask"][0],
            "label": torch.tensor(float(record["label"]), dtype=torch.float32),
        }


def load_records(path, image_root=None):
    """Load JSON/JSONL/CSV records.

    Required fields: image_path, title, true_reason, false_reason, label.
    Optional field: event_id.
    """
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif path.suffix.lower() == ".json":
        records = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            records = list(csv.DictReader(stream))
    else:
        raise ValueError(f"Unsupported record file: {path}")

    image_root = Path(image_root) if image_root else None
    normalized = []
    for item in records:
        item = dict(item)
        image_path = Path(item["image_path"])
        if image_root and not image_path.is_absolute():
            image_path = image_root / image_path
        item["image_path"] = str(image_path)
        item["label"] = int(item["label"])
        normalized.append(item)
    return normalized


def event_disjoint_split(records, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Split by event_id; this avoids event overlap between train/val/test."""
    groups = defaultdict(list)
    for record in records:
        groups[str(record.get("event_id", record.get("id", record["image_path"])))].append(record)

    events = list(groups)
    rng = random.Random(seed)
    rng.shuffle(events)
    n_train = int(len(events) * train_ratio)
    n_val = int(len(events) * val_ratio)
    train_events = events[:n_train]
    val_events = events[n_train:n_train + n_val]
    test_events = events[n_train + n_val:]

    def flatten(names):
        return [row for name in names for row in groups[name]]

    return flatten(train_events), flatten(val_events), flatten(test_events)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
