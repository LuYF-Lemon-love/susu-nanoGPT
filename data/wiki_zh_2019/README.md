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



after running `prepare.py` (preprocess) we get:

- train.bin is ~17GB, val.bin ~8.5MB
- train has ~9B tokens (9,035,582,198)
- val has ~4M tokens (4,434,897)

this came from 8,013,769 documents in total.

## 参考

[1] [nlp_chinese_corpus](https://github.com/brightmart/nlp_chinese_corpus)

[2] [json文件和jsonl文件有什么区别？](https://blog.csdn.net/Backli/article/details/131554069)

[3] [uer/gpt2-chinese-cluecorpussmall](https://huggingface.co/uer/gpt2-chinese-cluecorpussmall)

[4] [tokenizers/quicktour](https://huggingface.co/docs/tokenizers/quicktour)

[5] [transformers/fast_tokenizers](https://huggingface.co/docs/transformers/fast_tokenizers)
