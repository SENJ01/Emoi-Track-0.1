import os
import copy
import json
import logging
import torch
from torch.utils.data import TensorDataset
import pandas as pd
from pathlib import Path


logger = logging.getLogger(__name__)


class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, labels=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.labels = labels or []


class GoEmotionsProcessor(object):
    """Processor for GoEmotions / negative emotions dataset."""

    def __init__(self, args):
        self.args = args
        self.base_dir = Path(__file__).resolve().parents[2]

    def get_labels(self):
        labels = []
        label_file = Path(self.args.data_dir) / self.args.label_file
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                labels.append(line.strip())
        return labels

    def _create_negative_examples(self, csv_file, set_type):
        df = pd.read_csv(csv_file)
        examples = []
        for i, row in df.iterrows():
            guid = f"{set_type}-{i}"
            text_a = row["text"]
            label_list = row["label"].split(",")
            examples.append(InputExample(guid=guid, text_a=text_a, labels=label_list))
        return examples

    def get_examples(self, mode):
        file_path = getattr(self.args, f"{mode}_file")
        if getattr(self.args, "task", "") in ["negative_emo", "emopillars_negative"]:
            return self._create_negative_examples(
                os.path.join(self.args.data_dir, file_path), mode
            )
        else:
            with open(
                os.path.join(self.args.data_dir, file_path), "r", encoding="utf-8"
            ) as f:
                lines = f.readlines()
            examples = []
            for i, line in enumerate(lines):
                guid = f"{mode}-{i}"
                items = line.strip().split("\t")
                text_a = items[0]
                labels = list(map(int, items[1].split(",")))
                examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
            return examples


def convert_examples_to_features(examples, label_list, max_seq_len, tokenizer):
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []

    # Convert each processed example into model-ready tokenized features
    for example in examples:
        batch_encoding = tokenizer(
            example.text_a,
            max_length=max_seq_len,  # enforce the configured maximum sequence length
            padding="max_length",  # pad shorter sequences
            truncation=True,  # truncate longer sequences
            return_tensors="pt",
        )

        # Extract the token IDs and attention mask
        input_ids = batch_encoding["input_ids"][0]
        attention_mask = batch_encoding["attention_mask"][0]

        token_type_ids = batch_encoding.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids[0]
        else:
            token_type_ids = None

        # Convert labels into one-hot format for training
        label_ids = [0] * len(label_list)
        for label in example.labels:
            label_ids[label_map[label]] = 1

        features.append(
            (
                input_ids,
                attention_mask,
                token_type_ids,
                torch.tensor(label_ids, dtype=torch.float),
            )
        )

    return features


def load_and_cache_examples(args, tokenizer, mode):
    processor = GoEmotionsProcessor(args)
    cached_file = Path(args.data_dir) / (
        f"cached_{args.task}_{args.model_name_or_path.replace('/', '-')}_{args.max_seq_len}_{mode}.pt"
    )

    if cached_file.exists():
        logger.info(f"Loading features from cached file {cached_file}")
        features = torch.load(cached_file)
    else:
        logger.info(f"Creating features from dataset file at {args.data_dir}")
        examples = processor.get_examples(mode)
        features = convert_examples_to_features(
            examples, processor.get_labels(), args.max_seq_len, tokenizer
        )
        logger.info(f"Saving features into cached file {cached_file}")
        torch.save(features, cached_file)

    all_input_ids = torch.stack([f[0] for f in features])
    all_attention_mask = torch.stack([f[1] for f in features])

    if features[0][2] is not None:
        all_token_type_ids = torch.stack([f[2] for f in features])
    else:
        all_token_type_ids = torch.zeros_like(all_input_ids)

    all_labels = torch.stack([f[3] for f in features])

    return TensorDataset(
        all_input_ids, all_attention_mask, all_token_type_ids, all_labels
    )
