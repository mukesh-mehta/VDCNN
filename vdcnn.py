"""Character-level PyTorch VDCNN with configurable depth and a CPU training CLI."""
import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import string

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ALPHABET = string.ascii_lowercase + string.digits + string.punctuation + " "
CHAR_IDS = {char: i + 2 for i, char in enumerate(ALPHABET)}  # PAD=0, UNK=1
DEPTHS = {9: (1, 1, 1, 1), 17: (2, 2, 2, 2), 29: (5, 5, 2, 2), 49: (8, 8, 5, 3)}


def encode(texts, length=256):
    if length < 8 or not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Supply nonempty strings and a sequence length of at least 8")
    result = torch.zeros((len(texts), length), dtype=torch.long)
    for row, text in enumerate(texts):
        values = [CHAR_IDS.get(char, 1) for char in text.lower()[:length]]
        result[row, :len(values)] = torch.tensor(values)
    return result


class KMaxPool(nn.Module):
    """Select k highest activations while preserving their original temporal order."""
    def __init__(self, k):
        super().__init__()
        if k < 1:
            raise ValueError("k must be positive")
        self.k = k

    def forward(self, values):
        if values.shape[-1] < self.k:
            raise ValueError("Sequence is too short for k-max pooling; increase length or reduce k")
        indices = values.topk(self.k, dim=-1).indices.sort(dim=-1).values
        return values.gather(-1, indices)


@dataclass
class Config:
    classes: int = 2
    depth: int = 9
    width: int = 64
    embedding: int = 16
    k: int = 8
    hidden: int = 2048


class VDCNN(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or Config()
        c = self.config
        if c.depth not in DEPTHS or min(c.classes, c.width, c.embedding, c.k, c.hidden) < 1:
            raise ValueError("Unsupported depth or nonpositive model dimension")
        self.embedding = nn.Embedding(len(CHAR_IDS) + 2, c.embedding, padding_idx=0)
        layers = [nn.Conv1d(c.embedding, c.width, kernel_size=3, padding=1)]
        channels = c.width
        for stage, blocks in enumerate(DEPTHS[c.depth]):
            output = c.width * 2 ** stage
            for _ in range(blocks):
                for _ in range(2):
                    layers.extend([nn.Conv1d(channels, output, 3, padding=1),
                                   nn.BatchNorm1d(output), nn.ReLU()])
                    channels = output
            if stage < 3:
                layers.append(nn.MaxPool1d(3, stride=2, padding=1))
        self.convolutions = nn.Sequential(*layers)
        self.pool = KMaxPool(c.k)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(channels * c.k, c.hidden), nn.ReLU(),
                                  nn.Linear(c.hidden, c.hidden), nn.ReLU(), nn.Linear(c.hidden, c.classes))

    def forward(self, tokens):
        if tokens.ndim != 2 or tokens.shape[1] < 8 * (self.config.k - 1) + 1:
            raise ValueError("Expected [batch, sequence] tokens long enough for three pooling stages")
        return self.head(self.pool(self.convolutions(self.embedding(tokens).transpose(1, 2))))


def save(model, path, length):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects a previous training run from accidental overwrite.
    with path.open("xb") as stream:
        torch.save({"format_version": 1, "config": asdict(model.config), "length": length,
                    "alphabet": ALPHABET, "state_dict": model.state_dict()}, stream)


def load(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["format_version"] != 1 or checkpoint["alphabet"] != ALPHABET:
        raise ValueError("Unsupported checkpoint format or character vocabulary")
    model = VDCNN(Config(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), checkpoint["length"]


def train(texts, labels, *, epochs=3, batch_size=16, length=256, seed=42, config=None):
    if epochs < 1 or batch_size < 1 or len(texts) != len(labels) or len(texts) < 2:
        raise ValueError("Require at least two aligned examples and positive epochs/batch size")
    if any(not isinstance(label, int) or label < 0 for label in labels):
        raise ValueError("Labels must be nonnegative integers")
    classes = sorted(set(labels))
    if classes != list(range(len(classes))) or len(classes) < 2:
        raise ValueError("Use contiguous class IDs starting at 0, with at least two classes")
    config = config or Config(classes=len(classes))
    if config.classes != len(classes):
        raise ValueError("Config classes must match training labels")
    torch.manual_seed(seed)
    model = VDCNN(config)
    dataset = TensorDataset(encode(texts, length), torch.tensor(labels))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    for _ in range(epochs):
        model.train()
        total = 0.0
        for tokens, targets in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(tokens), targets)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item() * len(targets)
        history.append(total / len(dataset))
    return model.eval(), history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=2)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("csv", type=Path)
    training.add_argument("checkpoint", type=Path)
    training.add_argument("--epochs", type=int, default=3)
    training.add_argument("--length", type=int, default=256)
    training.add_argument("--depth", type=int, choices=DEPTHS, default=9)
    training.add_argument("--tiny", action="store_true", help="Use width=8, hidden=64 for CPU experiments")
    prediction = commands.add_parser("predict")
    prediction.add_argument("checkpoint", type=Path)
    prediction.add_argument("text", nargs="+")
    args = parser.parse_args()
    try:
        if args.threads < 1:
            raise ValueError("threads must be positive")
        torch.set_num_threads(args.threads)
        if args.command == "train":
            with args.csv.open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            texts, labels = [r["text"] for r in rows], [int(r["label"]) for r in rows]
            config = Config(classes=len(set(labels)), depth=args.depth,
                            width=8 if args.tiny else 64, hidden=64 if args.tiny else 2048)
            model, losses = train(texts, labels, epochs=args.epochs, length=args.length, config=config)
            save(model, args.checkpoint, args.length)
            print(json.dumps({"training_loss": losses, "examples": len(rows), "checkpoint": str(args.checkpoint)}))
        else:
            model, length = load(args.checkpoint)
            with torch.inference_mode():
                probabilities = model(encode(args.text, length)).softmax(dim=1)
            print(json.dumps([{"class_id": int(p.argmax()), "probabilities": p.tolist()} for p in probabilities]))
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
