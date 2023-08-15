
# susu-nanoGPT

## 安装

```shell
$ git clone git@github.com:LuYF-Lemon-love/susu-nanoGPT.git
$ cd susu-nanoGPT
$ python -m venv env
$ source env/bin/activate
$ which python
$ pip install --upgrade pip
$ pip install torch numpy transformers datasets tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 抽样 / 推断

使用脚本 `sample.py` 可以从 OpenAI 发布的预先训练的 GPT-2 模型中进行采样，也可以从您自己训练的模型中采样。例如，这里有一种从最大的可用 `gpt2-xl` 模型中采样的方法：

```
$ python sample.py \
    --init_from=gpt2-xl \
    --start="What is the answer to life, the universe, and everything?" \
    --num_samples=5 --max_new_tokens=100
```

如果您想从您训练的模型中进行采样，请使用 `--out_dir` 适当地指向代码。您还可以用文件中的一些文本提示模型，例如 `$ python sample.py --start=FILE:prompt.txt`。

## 参考

[1] [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)

[2] [karpathy/minGPT](https://github.com/karpathy/minGPT)

[3] [OpenWebText](https://openwebtext2.readthedocs.io/en/latest/)：OpenAI 的（私有）WebText 的开放复制品

[4] [OpenWebText-HuggingFace](https://huggingface.co/datasets/Skylion007/openwebtext)：OpenAI 的（私有）WebText 的开放复制品

[5] [eval和ast.literal_eval方法](https://blog.csdn.net/sinat_33924041/article/details/88350569)

[6] [CUDA_VISIBLE_DEVICES作用](https://blog.csdn.net/pxm_wzs/article/details/127886259)

[7] [设置CUDA_VISIBLE_DEVICES的方法](https://blog.csdn.net/B_DATA_NUIST/article/details/107973053)

[8] [Python 和 C++ 内存映射](http://139.129.163.161/index/toolkits)

[9] [contextlib.nullcontext(enter_result=None)](https://docs.python.org/zh-cn/3/library/contextlib.html?highlight=nullcontext#contextlib.nullcontext)

[10] [docs.wandb.ai/quickstart](https://docs.wandb.ai/quickstart)

[11] [Typical Mixed Precision Training](https://pytorch.org/docs/stable/notes/amp_examples.html#amp-examples)

[12] [the official GPT-2 TensorFlow implementation released by OpenAI](https://github.com/openai/gpt-2/blob/master/src/model.py)

[13] [huggingface/transformers PyTorch GPT-2 implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py)

[14] [@classmethod](https://docs.python.org/zh-cn/3/library/functions.html?highlight=classmethod#classmethod)

[15] [@classmethod是什么意思？Python](https://blog.csdn.net/qq_33945243/article/details/129409350)

[16] [Implementing Custom Decoders?](https://github.com/huggingface/tokenizers/issues/636)

[17] [custom_components.py](https://github.com/huggingface/tokenizers/blob/9a93c50c25c1e0b73a85584f327113bcbef5ac80/bindings/python/examples/custom_components.py#L44)

[18] [OpenWebText-10k](https://huggingface.co/datasets/stas/openwebtext-10k)