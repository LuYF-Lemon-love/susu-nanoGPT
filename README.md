
# susu-nanoGPT

## 安装

```shell
$ git clone git@github.com:LuYF-Lemon-love/susu-nanoGPT.git
$ cd susu-nanoGPT
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

## 复现 GPT-2

我们首先标记数据集，在本例中为 [OpenWebText-10k](https://huggingface.co/datasets/stas/openwebtext-10k)：

```
$ python data/openwebtext/prepare.py
```

这将下载并标记 [OpenWebText-10k](https://huggingface.co/datasets/stas/openwebtext-10k) 数据集。它将创建一个 `train.bin` 和 `val.bin`，在一个序列中保存 GPT2 BPE id，存储为 uint16 字节。然后我们就可以开始训练了。

```
$ python train.py config/train_gpt2.py
```

生成示例：

```shell
$ python sample.py
```

## 微调

微调与训练没有什么不同，我们只是确保从预先训练的模型中初始化，并以较小的学习率进行训练。关于如何在新文本上微调 `GPT` 的示例，请转到 `data/shakespeare` 并运行 `prepare.py` 下载小的 `shakespeare` 数据集，并使用 `GPT-2` 中的 OpenAI BPE 标记器将其呈现为 `train.bin` 和 `val.bin` 。

```
$ python data/shakespeare/prepare.py
$ python train.py config/finetune_shakespeare.py
```

基本上，我们使用 `init_from` 从 GPT2 检查点进行初始化，并像往常一样进行训练，只是时间更短，学习率很低。如果内存不足，请尝试减小模型大小（它们是 `{'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}`），或者可能减小 `block_size`（上下文长度）。根据配置文件，最好的检查点（最低的验证损失）将保存在 `out_dir` 目录中，例如默认情况下在 `out-shakespeare` 中。然后，您可以运行代码  `python sample.py --out_dir=out-shakespeare`：

```
THEODORUS:
O hush, you have nought to say of your father's war.

HIPPOCENTES:
O, but I will tell you the reason thereof.

THEODORUS:
O thou, Hippocrates, what didst thou answer him?

HIPPOCENTES:
O, when he had so much as said, to hold his peace.

THEODORUS:
O, come home: I will trouble you not to hear all that I say.

HIPPOCENTES:
O, to hear it, and to take it well.
```

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

## 参考

[1] [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)

[2] [karpathy/minGPT](https://github.com/karpathy/minGPT)

[3] [OpenWebText](https://openwebtext2.readthedocs.io/en/latest/)：OpenAI 的（私有）WebText 的开放复制品

[4] [OpenWebText-HuggingFace](https://huggingface.co/datasets/Skylion007/openwebtext)：OpenAI 的（私有）WebText 的开放复制品