from transformers import AutoConfig,AutoModelForMaskedLM,AutoTokenizer
from transformers import TrainingArguments,Trainer,HfArgumentParser,set_seed,PreTrainedTokenizer
import numpy as np
import pickle
from dataclasses import dataclass, field
from transformers.trainer_utils import is_main_process
from typing import List
from dataset import CorpusDataset
from transformers.data import DataCollatorForLanguageModeling
@dataclass
class DataTrainingArguments:
    data_path: str = field()
    max_seq_length: int = field()
@dataclass
class ModelArguments:
    model_name_or_path: str = field()
def read_pkl(datapath):
    dataset = []
    with open(datapath,'rb') as f :
        data = pickle.load(f)
    for idx, line in enumerate(data):
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        splits = line.split("\t")
        if len(splits) == 2:
            _id, text = splits
        else:
            raise NotImplementedError("Corpus Format: id\\ttext\\n")
        dataset.append(text)
    return dataset

def main():
    parser = HfArgumentParser(DataTrainingArguments,ModelArguments,TrainingArguments)
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, config=config, use_fast=False)
    model = AutoModelForMaskedLM.from_pretrained(model_args.model_name_or_path, config=config)
    train_set = CorpusDataset(
        read_pkl(data_args.corpus_path),
        tokenizer, data_args.max_seq_length
    )
    data_loader = DataCollatorForLanguageModeling(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        data_collator=data_loader,
        args=training_args,
        train_dataset=train_set
    )
    trainer.train()
    if is_main_process(training_args.local_rank):
        trainer.save_model()
if __name__ == "__main__":
    main()
