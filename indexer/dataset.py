import numpy as np
import random
from dataclasses import dataclass, field
from torch.utils.data import Dataset
from typing import List
from transformers import PreTrainedTokenizer
class CorpusDataset(Dataset):
    def __init__(self, corpus: List[str], tokenizer: PreTrainedTokenizer, max_seq_length: int):
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.max_token_len = max_seq_length - 2

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, item):
        if isinstance(self.corpus[item], str):
            text = self.corpus[item]
            cache_input_ids = self.tokenizer(text, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)['input_ids']
            self.corpus[item] = np.array(cache_input_ids, dtype=np.uint16) # cache
        
        input_ids = self.corpus[item].tolist()
        if len(input_ids) > self.max_token_len:
            start_pos = random.randint(0, len(input_ids)-self.max_token_len)
            input_ids = input_ids[start_pos: start_pos + self.max_token_len]
        batch_encoding = self.tokenizer.prepare_for_model(input_ids, add_special_tokens=True, return_special_tokens_mask=True)
        return batch_encoding