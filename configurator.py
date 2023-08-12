# coding:utf-8
#
# configurator.py
#
# git pull from nanoGPT by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on August 10, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on August 12, 2023
#
# 字符级 GPT 的数据预处理.

"""
该文件中的代码将从 train.py 运行如下：

>>> exec(open('configurator.py').read())

然后，此脚本中的代码将覆盖 globals()
"""

import sys
from ast import literal_eval

for arg in sys.argv[1:]:
    if '=' not in arg:
        # assume it's the name of a config file
        assert not arg.startswith('--')
        config_file = arg
        print(f"Overriding config with {config_file}:")
        with open(config_file) as f:
            print(f.read())
        exec(open(config_file).read())
    else:
        # assume it's a --key=value argument
        assert arg.startswith('--')
        key, val = arg.split('=')
        key = key[2:]
        if key in globals():
            try:
                # attempt to eval it it (e.g. if bool, number, or etc)
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                # if that goes wrong, just use the string
                attempt = val
            # ensure the types match ok
            assert type(attempt) == type(globals()[key])
            # cross fingers
            print(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")
