import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from tqdm import tqdm

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    # TODO: crecate test dataset
    with open(args.test_file, "r") as f:
        test_data = json.load(f)
    test_dataset = SeqTaggingClsDataset(test_data, vocab, tag2idx, args.max_len)
    # TODO: create DataLoader for test dataset
    test_loader = torch.utils.data.DataLoader(
        test_dataset, shuffle=False, batch_size=150, collate_fn=test_dataset.collate_fn
    )

    ckpt = torch.load(args.ckpt_path) if hasattr(args, "ckpt_path") else torch.load(args.ckpt_dir / args.ckpt)
    # load weights into model
    # TODO: implement main function
    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    model = SeqTagger(
        embeddings=embeddings,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
        num_class=10,
        batch_size=args.batch_size,
    )
    if ("cuda" not in args.device.type) and torch.cuda.is_available():
        args.device = torch.device("cuda:0")
    device = args.device
    print(f"using device {device}")
    model.to(device)
    model.load_state_dict(ckpt)
    model.eval()

    # TODO: predict dataset
    # TODO: write prediction to file (args.pred_file)
    ids = [td["id"] for td in test_data]
    with open(args.pred_file, "w") as f:
        f.write("id,tags\n")
        with torch.no_grad():
            for i, test in enumerate(tqdm(test_loader)):
                out = model(test["tokens"].to(args.device))
                _, preds = torch.max(out, 2)
                for j, pred in enumerate(preds):
                    pred_not_9 = [_ for _ in pred.tolist() if _ != 9]
                    label_not_9 = [test_dataset.idx2label(_) for _ in pred_not_9]
                    f.write(f"{ids[150*i+j]},{' '.join(label_not_9)}\n")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--test_file", type=Path, help="Path to the test file.", required=True)
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
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/kaggle.ckpt",
    )
    parser.add_argument("--pred_file", type=Path, default="pred.slot.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument("--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
