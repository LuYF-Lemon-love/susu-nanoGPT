# -----------------------------------------------------------------------------
# 默认配置
# I/O
eval_interval = 100
log_interval = 10
always_save_checkpoint = True
# wandb logging
wandb_log = False
wandb_project = 'wiki-zh'
wandb_run_name = 'gpt2'
# data
dataset = 'wiki_zh_2019'
# adamw optimizer
learning_rate = 6e-3
max_iters = 50000
# learning rate decay settings
lr_decay_iters = 50000
min_lr = 6e-4