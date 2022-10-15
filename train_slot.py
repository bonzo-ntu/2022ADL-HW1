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

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
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


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        help="Directory to save the model file.",
        default="model.ckpt",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument("--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu")
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
