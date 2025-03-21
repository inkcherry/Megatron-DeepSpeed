# Utilizing code snippet from https://github.com/tatsu-lab/stanford_alpaca
import copy
import logging
from typing import Dict, Sequence
import io
import torch
import transformers
from torch.utils.data import Dataset
import json
import tqdm
import os
import pickle

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}
def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

DATA_CACHE_SUFFIX=".temp"

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer, seq_len):
        logging.warning(f"using {tokenizer.__class__.__name__}")
        if tokenizer.__class__.__name__ == "WeLMV3Tokenizer":
            tokenizer = tokenizer
        else:
            tokenizer = tokenizer.tokenizer
        super(SupervisedDataset, self).__init__()
        dataset_cache_filename = data_path + tokenizer.__class__.__name__ + str(seq_len) + DATA_CACHE_SUFFIX
        #if os.path.exists(dataset_cache_filename):
        if False:
            with open(dataset_cache_filename, 'rb') as f:
                data_dict = pickle.load(f) 
            logging.warning("loaded data cache from ", dataset_cache_filename)
        else:
            logging.warning("Loading data...")
            list_data_dict = jload(data_path)
            logging.warning("Formatting inputs...")
            prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
            sources = [
                prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
                for example in list_data_dict
            ]
            targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

            logging.warning("Tokenizing inputs... This may take some time...")
            data_dict = preprocess(sources, targets, tokenizer, seq_len)
            #with open(dataset_cache_filename, 'wb') as f:
            #    pickle.dump(data_dict, f)
            #logging.warning("saved data cache to ", dataset_cache_filename)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer, seq_len) -> Dict:
    """Tokenize a list of strings."""
    # +1 for alignment labels and tokens
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=seq_len + 1,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    seq_len: int,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer, seq_len) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)

    "Here we use padding to fill the prompt in the labels."
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = tokenizer.pad_token_id
    return dict(input_ids=input_ids, labels=labels)
