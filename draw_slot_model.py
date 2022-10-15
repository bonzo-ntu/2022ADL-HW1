import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from tqdm import trange, tqdm

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab
from torch import nn


class args:
    data_dir = Path("./data/slot/")
    cache_dir = Path("./cache/slot/")
    ckpt_dir = Path("./ckpt/slot/")
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

tag_idx_path = args.cache_dir / "tag2idx.json"
tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
datasets: Dict[str, SeqTaggingClsDataset] = {
    split: SeqTaggingClsDataset(split_data, vocab, tag2idx, args.max_len) for split, split_data in data.items()
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
model = SeqTagger(
    embeddings=embeddings,
    hidden_size=args.hidden_size,
    num_layers=args.num_layers,
    dropout=args.dropout,
    bidirectional=args.bidirectional,
    num_class=10,
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
        inputs, labels = data["tokens"].to(device), data["tags"].to(device)
        optimizer.zero_grad()
        out = model(inputs)
        loss = loss_func(out.view(-1, 10), labels.view(-1))
        _, train_pred = torch.max(out, 2)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        for j, label in enumerate(labels):
            train_acc += (train_pred[j].cpu() == label.cpu()).sum().item() == 64
    else:
        train_loss /= len(train_loader)
        train_acc /= len(train_loader.dataset)

    # TODO: Evaluation loop - calculate accuracy and save model weights
    with torch.no_grad():
        model.eval()
        for dev_data in tqdm(dev_loader):
            inputs, labels = dev_data["tokens"].to(device), dev_data["tags"].to(device)
            out = model(inputs)
            loss = loss_func(out.view(-1, 10), labels.view(-1))
            _, val_pred = torch.max(out, 2)
            valid_loss += loss.item()

            for j, label in enumerate(labels):
                valid_acc += (val_pred[j].cpu() == label.cpu()).sum().item() == 64
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


out = model(inputs)


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


from torchviz import make_dot

dot = make_dot(out.mean(), params=dict(model.named_parameters()))
resize_graph(dot, size_per_element=1, min_size=20)
dot.render("slot", format="png")
