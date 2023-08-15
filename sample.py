# coding:utf-8
#
# sample.py
#
# git pull from nanoGPT by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on August 10, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on August 14, 2023
#
# GPT 生成样本.

import os
from contextlib import nullcontext
import torch
from tokenizers import Tokenizer
from tokenizers.decoders import Decoder
from typing import List
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
out_dir = 'out'
start = "生命、宇宙和一切的答案是什么？"
num_samples = 10
max_new_tokens = 500
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200
seed = 42
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False
exec(open('configurator.py').read())
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
model.to(device)
if compile:
    model = torch.compile(model)

print("加载分词器...")
class CustomDecoder:
    def decode_chain(self, tokens: List[str]) -> List[str]:
        return [f"{t}" for t in tokens]
tokenizer = Tokenizer.from_file(os.path.join('data', checkpoint['config']['dataset'], "data/wiki_zh_2019/tokenizer.json"))
tokenizer.decoder = Decoder.custom(CustomDecoder())
encode = lambda s: tokenizer.encode(s).ids
decode = lambda l: tokenizer.decode(l)

if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')
