import json
import os
import numpy as np
from typing import List

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from download import download
import pandas as pd


import csv
from tqdm import tqdm

def load_texts_from_csv(csv_file):
    texts = []

    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc=f'Loading {csv_file}'):
            texts.append(row['text'])  # Assuming the CSV has a column named 'text'

    return texts


class Corpus:
    def __init__(self, name, data_dir='data', skip_train=False):
        self.name = name
        self.train = load_texts_from_csv(f'{data_dir}/{name}.train.csv') if not skip_train else None
        self.test = load_texts_from_csv(os.path.join(data_dir, f'{name}.test.csv'))
        self.valid = load_texts_from_csv(f'{data_dir}/{name}.valid.csv')
        if skip_train:
            self.valid.extend(self.test)

class EncodedDataset(Dataset):
    def __init__(self, real_texts: List[str], fake_texts: List[str], tokenizer: PreTrainedTokenizer,
                 max_sequence_length: int = None, min_sequence_length: int = None, epoch_size: int = None,
                 token_dropout: float = None, seed: int = None):
        self.real_texts = real_texts
        self.fake_texts = fake_texts
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.min_sequence_length = min_sequence_length
        self.epoch_size = epoch_size
        self.token_dropout = token_dropout
        self.random = np.random.RandomState(seed)

    def __len__(self):
        return self.epoch_size or len(self.real_texts) + len(self.fake_texts)

    def __getitem__(self, index):
        if self.epoch_size is not None:
            label = self.random.randint(2)
            texts = [self.fake_texts, self.real_texts][label]
            text = texts[self.random.randint(len(texts))]
        else:
            if index < len(self.real_texts):
                text = self.real_texts[index]
                label = 1
            else:
                text = self.fake_texts[index - len(self.real_texts)]
                label = 0

        tokens = self.tokenizer.encode(text)

        if self.max_sequence_length is None:
            tokens = tokens[:self.tokenizer.model_max_length - 2]
        else:
            output_length = min(len(tokens), self.model_max_length)
            if self.min_sequence_length:
                output_length = self.random.randint(min(self.min_sequence_length, len(tokens)), output_length + 1)
            start_index = 0 if len(tokens) <= output_length else self.random.randint(0, len(tokens) - output_length + 1)
            end_index = start_index + output_length
            tokens = tokens[start_index:end_index]

        if self.token_dropout:
            dropout_mask = self.random.binomial(1, self.token_dropout, len(tokens)).astype(np.bool)
            tokens = np.array(tokens)
            tokens[dropout_mask] = self.tokenizer.unk_token_id
            tokens = tokens.tolist()

        if self.max_sequence_length is None or len(tokens) == self.max_sequence_length:
            mask = torch.ones(len(tokens) + 2)
            return torch.tensor([self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]), mask, label

        padding = [self.tokenizer.pad_token_id] * (self.max_sequence_length - len(tokens))
        tokens = torch.tensor([self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id] + padding)
        mask = torch.ones(tokens.shape[0])
        mask[-len(padding):] = 0
        return tokens, mask, label
    
class TuringBenchDataset(Dataset):
    def __init__(self, csv_file, tokenizer: PreTrainedTokenizer,
                 max_sequence_length: int = None, min_sequence_length: int = None, epoch_size: int = None,
                 token_dropout: float = None, seed: int = None):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.min_sequence_length = min_sequence_length
        self.epoch_size = epoch_size
        self.token_dropout = token_dropout
        self.random = np.random.RandomState(seed)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.epoch_size is not None:
            label = self.random.randint(2)
            texts = [self.fake_texts, self.real_texts][label]
            text = texts[self.random.randint(len(texts))]
        else:
            text = self.data.iloc[index]["Generation"]
            label = self.data.iloc[index]["label"]
            label = 'human' in label

        tokens = self.tokenizer.encode(text)

        if self.max_sequence_length is None:
            tokens = tokens[:self.tokenizer.model_max_length - 2]
        else:
            output_length = min(len(tokens), self.model_max_length)
            if self.min_sequence_length:
                output_length = self.random.randint(min(self.min_sequence_length, len(tokens)), output_length + 1)
            start_index = 0 if len(tokens) <= output_length else self.random.randint(0, len(tokens) - output_length + 1)
            end_index = start_index + output_length
            tokens = tokens[start_index:end_index]

        if self.token_dropout:
            dropout_mask = self.random.binomial(1, self.token_dropout, len(tokens)).astype(np.bool)
            tokens = np.array(tokens)
            tokens[dropout_mask] = self.tokenizer.unk_token_id
            tokens = tokens.tolist()

        if self.max_sequence_length is None or len(tokens) == self.max_sequence_length:
            mask = torch.ones(len(tokens) + 2)
            return torch.tensor([self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]), mask, label

        padding = [self.tokenizer.pad_token_id] * (self.max_sequence_length - len(tokens))
        tokens = torch.tensor([self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id] + padding)
        mask = torch.ones(tokens.shape[0])
        mask[-len(padding):] = 0
        return tokens, mask, label
