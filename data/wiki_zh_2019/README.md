# 维基百科(wiki2019zh)，100万个结构良好的中文词条

数据集为: [维基百科中文数据集](https://github.com/brightmart/nlp_chinese_corpus)包括 100 万个结构良好的中文词条（1,043,224 条数据）。

1. 解压数据集：

```shell
$ cd data/wiki_zh_2019/
$ unzip wiki_zh_2019.zip
```

2. 安装 jupyter:

```shell
$ pip install jupyter -i https://pypi.tuna.tsinghua.edu.cn/simple
```

3. 运行脚本 `json2text.ipynb` 处理数据并训练分词器，需要 60G 内存。

4. 运行 `python prepare.py` 将维基百科中文数据集保存到二进制文件中用于训练。

- train.bin: ~536M

- val.bin: ~2.5M

## 参考

[1] [nlp_chinese_corpus](https://github.com/brightmart/nlp_chinese_corpus)

[2] [json文件和jsonl文件有什么区别？](https://blog.csdn.net/Backli/article/details/131554069)

[3] [uer/gpt2-chinese-cluecorpussmall](https://huggingface.co/uer/gpt2-chinese-cluecorpussmall)

[4] [tokenizers/quicktour](https://huggingface.co/docs/tokenizers/quicktour)

[5] [transformers/fast_tokenizers](https://huggingface.co/docs/transformers/fast_tokenizers)

[6] [gpt2 简单示例](https://zhuanlan.zhihu.com/p/625791719)

[7] [生成模型特殊标记含义](https://blog.csdn.net/qq_37356556/article/details/131103015)

[8] [BERT和ERNIE中[PAD],[CLS],[SEP],[MASK],[UNK]所代表的含义](https://blog.csdn.net/weixin_43220532/article/details/124248411)

[9] [CLUECorpus2020](https://github.com/CLUEbenchmark/CLUECorpus2020/)

[10] [CLUE benchmark](https://github.com/CLUEbenchmark/CLUE)
