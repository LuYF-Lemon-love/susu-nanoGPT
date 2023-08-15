# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-wiki-zh'
eval_interval = 1000 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True # override via command line if you like
wandb_project = 'wiki-zh'
wandb_run_name = 'mini-gpt'

dataset = 'wiki_zh_2019'
gradient_accumulation_steps = 5 * 8
batch_size = 24
block_size = 512 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 8
n_head = 8
n_embd = 384
dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 300000
lr_decay_iters = 300000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially