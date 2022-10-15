import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from tqdm import trange, tqdm

from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab
from torch import nn

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


class args:
    data_dir = Path("./data/intent/")
    cache_dir = Path("./cache/intent/")
    ckpt_dir = Path("./ckpt/intent/")
    ckpt = Path("model.ckpt")
    max_len = int(128)
    hidden_size = int(512)
    dropout = float(0.1)
    bidirectional = bool(True)
    lr = float(1e-3)
    batch_size = int(128)
    device = torch.device("cuda:0")
    num_epoch = int(1)
    num_layers = int(2)


with open(args.cache_dir / "vocab.pkl", "rb") as f:
    vocab: Vocab = pickle.load(f)

intent_idx_path = args.cache_dir / "intent2idx.json"
intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
datasets: Dict[str, SeqClsDataset] = {
    split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len) for split, split_data in data.items()
}
# TODO: create DataLoader for train / dev datasets
train_loader = torch.utils.data.DataLoader(
    datasets[TRAIN], batch_size=args.batch_size, shuffle=True, collate_fn=datasets[TRAIN].collate_fn
)
dev_loader = torch.utils.data.DataLoader(
    datasets[DEV], batch_size=args.batch_size, shuffle=False, collate_fn=datasets[DEV].collate_fn
)

embeddings = torch.load(args.cache_dir / "embeddings.pt")
# TODO: init model and move model to target device(cpu / gpu)
model = SeqClassifier(
    embeddings=embeddings,
    hidden_size=args.hidden_size,
    num_layers=args.num_layers,
    dropout=args.dropout,
    bidirectional=args.bidirectional,
    num_class=150,
    batch_size=args.batch_size,
)

if ("cuda" not in args.device.type) and torch.cuda.is_available():
    args.device = torch.device("cuda:0")
device = args.device
print(f"using device {device}")
model.to(device)

# TODO: init optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

loss_func = nn.CrossEntropyLoss()
best_acc = 0.0
for epoch in trange(args.num_epoch, desc="Epoch"):
    # TODO: Training loop - iterate over train dataloader and update model weights
    train_loss, train_acc = 0.0, 0.0
    valid_loss, valid_acc = 0.0, 0.0

    model.train()
    for data in tqdm(train_loader):
        inputs, labels = data["text"].to(device), data["intent"].to(device)
        optimizer.zero_grad()
        out = model(inputs)
        loss = loss_func(out, labels)
        _, train_pred = torch.max(out, 1)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += (train_pred == labels).sum().item()
    else:
        train_loss /= len(train_loader)
        train_acc /= len(train_loader.dataset)

    # TODO: Evaluation loop - calculate accuracy and save model weights
    with torch.no_grad():
        model.eval()
        for dev_data in tqdm(dev_loader):
            inputs, labels = dev_data["text"].to(device), dev_data["intent"].to(device)
            out = model(inputs)
            loss = loss_func(out, labels)
            _, val_pred = torch.max(out, 1)
            valid_loss += loss.item()
            valid_acc += (val_pred == labels).sum().item()
        else:
            valid_loss /= len(dev_loader)
            valid_acc /= len(dev_loader.dataset)

        print(
            f"Epoch {epoch + 1}: Train Acc: {train_acc}, Train Loss: {train_loss}, \
                Val Acc: {valid_acc}, Val Loss: {valid_loss}"
        )
        if valid_acc >= best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), args.ckpt_dir / args.ckpt)
            print(f"Save model with acc {valid_acc}")

# TODO: Inference on test set


def resize_graph(dot, size_per_element=0.15, min_size=12):
    """Resize the graph according to how much content it contains.
    Modify the graph in place.
    """
    # Get the approximate number of nodes and edges
    num_rows = len(dot.body)
    content_size = num_rows * size_per_element
    size = max(min_size, content_size)
    size_str = str(size) + "," + str(size)
    dot.graph_attr.update(size=size_str)


out = model(inputs)

from torchviz import make_dot

dot = make_dot(out.mean(), params=dict(model.named_parameters()))
resize_graph(dot, size_per_element=1, min_size=20)
dot.render("intent", format="png")
