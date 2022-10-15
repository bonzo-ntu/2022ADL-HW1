# Sample Code for Homework 1 ADL NTU

## Train model
```shell
python train_intent.py --device cuda:0 --ckpt trained_intent.ckpt --dropout 0.25
python train_slot.py --device cuda:0 --ckpt trained_slot.ckpt --dropout 0.5
```

## Inference
```shell
bash ./intent_cls.sh ./data/intent/test.json pred_intent.csv
bash ./slot_tag.sh ./data/slot/test.json pred_slot.csv
```

## Environment
```shell
# If you have conda, we recommend you to build a conda environment called "adl-hw1"
make
conda activate adl-hw1
pip install -r requirements.txt
# Otherwise
pip install -r requirements.in
```

## Preprocessing
```shell
# To preprocess intent detectiona and slot tagging datasets
bash preprocess.sh
```

## Intent detection
```shell
python train_intent.py
```
