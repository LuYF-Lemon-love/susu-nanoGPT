# coding:utf-8
#
# data/openwebtext/prepare.py
#
# git pull from nanoGPT by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on August 10, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on August 13, 2023
#
# 中文 GPT 的数据预处理.

"""将维基百科中文数据集保存到二进制文件中用于训练。"""

import os
import json
import numpy as np
from tqdm import tqdm
from datasets import Dataset
from tokenizers import Tokenizer

num_proc = 64

def read_jsonl(jsonl_file):
    text_list = []
    with open(jsonl_file, "r") as file:
        for line in file:
            json_obj = json.loads(line)
            text_list.append(json_obj["text"].lstrip(json_obj["title"]).strip())
    return text_list

folder_path = 'wiki_zh'
text_list = []
for root, dirs, files in os.walk(folder_path):
    for file in files:
        file_path = os.path.join(root, file)
        text_list += read_jsonl(file_path)
print(f'一共{len(text_list)}条数据。')
print(text_list[0])

data_dict = {"text": text_list}
dataset = Dataset.from_dict(data_dict)
split_dataset = dataset.train_test_split(test_size=0.005, seed=42, shuffle=True)
split_dataset['val'] = split_dataset.pop('test')
print(split_dataset)

tokenizer = Tokenizer.from_file("tokenizer.json")
print(f"vocab_size: {tokenizer.get_vocab_size()}, [EOT]: {tokenizer.token_to_id('[EOT]')}")
print(split_dataset['train'][0])
print(tokenizer.encode(split_dataset['train'][0]['text']).tokens)
print(tokenizer.encode(split_dataset['train'][0]['text']).ids)

def process(example):
    ids = tokenizer.encode(example['text']).ids
    ids.append(tokenizer.token_to_id("[EOT]"))
    out = {'ids': ids, 'len': len(ids)}
    return out

tokenized = split_dataset.map(
    process,
    remove_columns=['text'],
    desc="tokenizing the splits",
    num_proc=num_proc,
)

# concatenate all the ids in each dataset into one large file we can use for training
for split, dset in tokenized.items():
    arr_len = np.sum(dset['len'], dtype=np.uint64)
    filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
    dtype = np.uint16
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
    total_batches = 32
    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
        # Batch together samples for faster write
        batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
        arr_batch = np.concatenate(batch['ids'])
        # Write into mmap
        arr[idx : idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)
    arr.flush()