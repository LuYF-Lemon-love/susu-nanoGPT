
# susu-nanoGPT

## 安装

```shell
$ python -m venv env
$ source env/bin/activate
$ which python
$ pip install --upgrade pip
$ pip install torch numpy transformers datasets tiktoken wandb tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 快速开始

最快的入门方法是在莎士比亚的作品中训练一个字符级的GPT。首先，我们将下载一个（1MB）文件，并将其从原始文本转换为一个大的整数流：

```shell
$ python data/shakespeare_char/prepare.py
```

这将在该数据目录中创建一个 `train.bin` 和 `val.bin`。现在是时候训练你的GPT了，配置文件为 [config/train_shakespeare_char.py](config/train_shakespeare_char.py)：

```
$ python train.py config/train_shakespeare_char.py
```

如果你仔细观察它，你会发现我们正在训练一个 GPT，其上下文大小最多为 256 个字符，384 个特征维度，它是一个 6 层的 Transformer，每层有 6 个头。根据参数 `out_dir` 模型保存在 `out-shakespeare-char`。因此，一旦训练完成，我们就可以通过将采样脚本指向以下目录来从最佳模型中采样：

```
$ python sample.py --out_dir=out-shakespeare-char
```

这会生成一些示例，例如：

```
Have he not set dropp'd her eyes and mours to another's
body? O enborn! I do not tell thee: and thou'rt
not scarcet upon him.

HORTENSIO:
Why, thou mayst prove the man gaspiness of hour it.

LUCIO:
Why should this it is, and there would die live?

MISTRESS OVERDONE:
Why, I pray you, sir, the unshares Margaret.

LUCIO:
Have you not changed to his bed-bed to prove me; and here
well hencefore you in the rash prisoners have to the ground.
```

 使用 `cpu` 训练 GPT:

```
$ python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```

在这里，由于我们在 CPU 而不是 GPU 上运行，我们必须同时设置 `--device=cpu`，并关闭 PyTorch 2.0 编译，同时使用 `--compile=False`。然后，当我们进行评估时，我们得到了一个更嘈杂但更快的估计（`-eval_iters=20`，原来是 `200` ），我们的上下文大小只有 `64` 个字符，而不是 `256` 个，并且每次迭代的批大小只有 `12` 个示例，而不是 `64` 个。我们还将使用更小的 `Transformer`（4 层，4 个头，128 嵌入大小），并将迭代次数减少到 `2000` 次（相应地，通常使用 `--lr_decay_iters` 将学习率降低到 `max_iters` 左右）。因为我们的网络太小了，所以我们也简化了正则化（`--dropout=0.0`）。因此也有更糟糕的样本，但它仍然很有趣：

```
$ python sample.py --out_dir=out-shakespeare-char --device=cpu
```

生成如下示例：

```
Let mother fater, for moir shot.

YORK:

Come you, bay my my as the arfort; though a proce,
I mone yous'd look somp.

DUKE VINCENNTIzA:
Hy shame for as the hort whoe head to more
And that anate quien
Gedieng all some the shant the bust thine
As the do may tough; I with incle swoll with not.
```

## reproducing GPT-2

A more serious deep learning professional may be more interested in reproducing GPT-2 results. So here we go - we first tokenize the dataset, in this case the [OpenWebText](https://openwebtext2.readthedocs.io/en/latest/), an open reproduction of OpenAI's (private) WebText:

```
$ python data/openwebtext/prepare.py
```

This downloads and tokenizes the [OpenWebText](https://huggingface.co/datasets/openwebtext) dataset. It will create a `train.bin` and `val.bin` which holds the GPT2 BPE token ids in one sequence, stored as raw uint16 bytes. Then we're ready to kick off training. To reproduce GPT-2 (124M) you'll want at least an 8X A100 40GB node and run:

```
$ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

This will run for about 4 days using PyTorch Distributed Data Parallel (DDP) and go down to loss of ~2.85. Now, a GPT-2 model just evaluated on OWT gets a val loss of about 3.11, but if you finetune it it will come down to ~2.85 territory (due to an apparent domain gap), making the two models ~match.

If you're in a cluster environment and you are blessed with multiple GPU nodes you can make GPU go brrrr e.g. across 2 nodes like:

```
Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
```

It is a good idea to benchmark your interconnect (e.g. iperf3). In particular, if you don't have Infiniband then also prepend `NCCL_IB_DISABLE=1` to the above launches. Your multinode training will work, but most likely _crawl_. By default checkpoints are periodically written to the `--out_dir`. We can sample from the model by simply `$ python sample.py`.

Finally, to train on a single GPU simply run the `$ python train.py` script. Have a look at all of its args, the script tries to be very readable, hackable and transparent. You'll most likely want to tune a number of those variables depending on your needs.

## baselines

OpenAI GPT-2 checkpoints allow us to get some baselines in place for openwebtext. We can get the numbers as follows:

```
$ python train.py eval_gpt2
$ python train.py eval_gpt2_medium
$ python train.py eval_gpt2_large
$ python train.py eval_gpt2_xl
```

and observe the following losses on train and val:

| model | params | train loss | val loss |
| ------| ------ | ---------- | -------- |
| gpt2 | 124M         | 3.11  | 3.12     |
| gpt2-medium | 350M  | 2.85  | 2.84     |
| gpt2-large | 774M   | 2.66  | 2.67     |
| gpt2-xl | 1558M     | 2.56  | 2.54     |

However, we have to note that GPT-2 was trained on (closed, never released) WebText, while OpenWebText is just a best-effort open reproduction of this dataset. This means there is a dataset domain gap. Indeed, taking the GPT-2 (124M) checkpoint and finetuning on OWT directly for a while reaches loss down to ~2.85. This then becomes the more appropriate baseline w.r.t. reproduction.

## finetuning

Finetuning is no different than training, we just make sure to initialize from a pretrained model and train with a smaller learning rate. For an example of how to finetune a GPT on new text go to `data/shakespeare` and run `prepare.py` to download the tiny shakespeare dataset and render it into a `train.bin` and `val.bin`, using the OpenAI BPE tokenizer from GPT-2. Unlike OpenWebText this will run in seconds. Finetuning can take very little time, e.g. on a single GPU just a few minutes. Run an example finetuning like:

```
$ python train.py config/finetune_shakespeare.py
```

This will load the config parameter overrides in `config/finetune_shakespeare.py` (I didn't tune them much though). Basically, we initialize from a GPT2 checkpoint with `init_from` and train as normal, except shorter and with a small learning rate. If you're running out of memory try decreasing the model size (they are `{'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}`) or possibly decreasing the `block_size` (context length). The best checkpoint (lowest validation loss) will be in the `out_dir` directory, e.g. in `out-shakespeare` by default, per the config file. You can then run the code in `sample.py --out_dir=out-shakespeare`:

```
THEODORE:
Thou shalt sell me to the highest bidder: if I die,
I sell thee to the first; if I go mad,
I sell thee to the second; if I
lie, I sell thee to the third; if I slay,
I sell thee to the fourth: so buy or sell,
I tell thee again, thou shalt not sell my
possession.

JULIET:
And if thou steal, thou shalt not sell thyself.

THEODORE:
I do not steal; I sell the stolen goods.

THEODORE:
Thou know'st not what thou sell'st; thou, a woman,
Thou art ever a victim, a thing of no worth:
Thou hast no right, no right, but to be sold.
```

Whoa there, GPT, entering some dark place over there. I didn't really tune the hyperparameters in the config too much, feel free to try!

## sampling / inference

Use the script `sample.py` to sample either from pre-trained GPT-2 models released by OpenAI, or from a model you trained yourself. For example, here is a way to sample from the largest available `gpt2-xl` model:

```
$ python sample.py \
    --init_from=gpt2-xl \
    --start="What is the answer to life, the universe, and everything?" \
    --num_samples=5 --max_new_tokens=100
```

If you'd like to sample from a model you trained, use the `--out_dir` to point the code appropriately. You can also prompt the model with some text from a file, e.g. `$ python sample.py --start=FILE:prompt.txt`.

## efficiency notes

For simple model benchmarking and profiling, `bench.py` might be useful. It's identical to what happens in the meat of the training loop of `train.py`, but omits much of the other complexities.

Note that the code by default uses [PyTorch 2.0](https://pytorch.org/get-started/pytorch-2.0/). At the time of writing (Dec 29, 2022) this makes `torch.compile()` available in the nightly release. The improvement from the one line of code is noticeable, e.g. cutting down iteration time from ~250ms / iter to 135ms / iter. Nice work PyTorch team!

## todos

- Investigate and add FSDP instead of DDP
- Eval zero-shot perplexities on standard evals (e.g. LAMBADA? HELM? etc.)
- Finetune the finetuning script, I think the hyperparams are not great
- Schedule for linear batch size increase during training
- Incorporate other embeddings (rotary, alibi)
- Separate out the optim buffers from model params in checkpoints I think
- Additional logging around network health (e.g. gradient clip events, magnitudes)
- Few more investigations around better init etc.

## troubleshooting

Note that by default this repo uses PyTorch 2.0 (i.e. `torch.compile`). This is fairly new and experimental, and not yet available on all platforms (e.g. Windows). If you're running into related error messages try to disable this by adding `--compile=False` flag. This will slow down the code but at least it will run.

For some context on this repository, GPT, and language modeling it might be helpful to watch my [Zero To Hero series](https://karpathy.ai/zero-to-hero.html). Specifically, the [GPT video](https://www.youtube.com/watch?v=kCc8FmEb1nY) is popular if you have some prior language modeling context.

For more questions/discussions feel free to stop by **#nanoGPT** on Discord:

[![](https://dcbadge.vercel.app/api/server/3zy8kqD9Cp?compact=true&style=flat)](https://discord.gg/3zy8kqD9Cp)

## acknowledgements

All nanoGPT experiments are powered by GPUs on [Lambda labs](https://lambdalabs.com), my favorite Cloud GPU provider. Thank you Lambda labs for sponsoring nanoGPT!

## 参考

[1] [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)

[2] [karpathy/minGPT](https://github.com/karpathy/minGPT)
