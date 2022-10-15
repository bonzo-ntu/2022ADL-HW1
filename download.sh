#!/bin/bash
# create the directories if they do not exist
mkdir -p ckpt/intent
mkdir -p ckpt/slot
mkdir -p data/intent
mkdir -p data/slot
mkdir -p cache/intent
mkdir -p cache/slot

# download cache files
# intent
embeddings_pt="https://www.dropbox.com/s/fcbpc1mn88d9z0h/embeddings.pt?dl=1"
intent2idx_json="https://www.dropbox.com/s/bxmail6draxnigb/intent2idx.json?dl=1"
vocab_pkl="https://www.dropbox.com/s/ssbeghvff39bnfc/vocab.pkl?dl=1"
wget ${embeddings_pt} -O ./cache/intent/embeddings.pt
wget ${intent2idx_json} -O ./cache/intent/intent2idx.json
wget ${vocab_pkl} -O ./cache/intent/vocab.pkl

# slot
embeddings_pt="https://www.dropbox.com/s/syg3243qapdt2hw/embeddings.pt?dl=1"
intent2idx_json="https://www.dropbox.com/s/8vok864p9xsz2y1/tag2idx.json?dl=1"
vocab_pkl="https://www.dropbox.com/s/w1gz2rd9dx5vnvz/vocab.pkl?dl=1"
wget ${embeddings_pt} -O ./cache/slot/embeddings.pt
wget ${intent2idx_json} -O ./cache/slot/tag2idx.json
wget ${vocab_pkl} -O ./cache/slot/vocab.pkl

# download data files
# intent
eval_json="https://www.dropbox.com/s/fki2dd75kqt0oei/eval.json?dl=1"
test_json="https://www.dropbox.com/s/qhzipzgejyfq0u2/test.json?dl=1"
train_json="https://www.dropbox.com/s/ttq8713ksq8wr6k/train.json?dl=1"
wget ${eval_json} -O ./data/intent/eval.json
wget ${test_json} -O ./data/intent/test.json
wget ${train_json} -O ./data/intent/train.json

# slot
eval_json="https://www.dropbox.com/s/9smxoby7rfdls7y/eval.json?dl=1"
test_json="https://www.dropbox.com/s/6wwxai31wdgzoon/test.json?dl=1"
train_json="https://www.dropbox.com/s/8cxfdyn47d726vc/train.json?dl=1"
wget ${eval_json} -O ./data/slot/eval.json
wget ${test_json} -O ./data/slot/test.json
wget ${train_json} -O ./data/slot/train.json

# download ckpt files
# intent
kaggle_ckpt="https://www.dropbox.com/s/5x8are4tyrlgtqj/kaggle.ckpt?dl=1"
wget ${kaggle_ckpt} -O ./ckpt/intent/kaggle.ckpt

# slot
kaggle_ckpt="https://www.dropbox.com/s/fh2yzmil46p65qs/kaggle.ckpt?dl=1"
wget ${kaggle_ckpt} -O ./ckpt/slot/kaggle.ckpt
