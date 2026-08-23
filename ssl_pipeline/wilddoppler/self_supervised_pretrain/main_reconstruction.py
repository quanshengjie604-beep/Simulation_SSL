from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from self_supervised_pretrain.data import make_loaders
from self_supervised_pretrain.trainers import BackboneReconstruction, ReconstructionTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DopplerWild reconstruction pretraining.")
    parser.add_argument("--tracklist-csv", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--ckpt-dir", default="self_supervised_pretrain/checkpoints/reconstruction")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--crop-seconds", type=float, default=1.0)
    parser.add_argument("--bins-per-second", type=int, default=90)
    parser.add_argument("--resize-doppler", type=int, default=256)
    parser.add_argument("--train-overlap-ratio", type=float, default=0.5)
    parser.add_argument("--uD-mean", type=float, default=15.589631)
    parser.add_argument("--uD-std", type=float, default=8.797207)
    parser.add_argument("--cache-mode", choices=["none", "lazy", "preload"], default="none")

    parser.add_argument("--model-name", choices=["mobilenet_v2", "resnet18"], default="mobilenet_v2")
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--stem-channels", type=int, default=32)
    parser.add_argument("--use-radar-stem", action="store_true")
    parser.add_argument("--pretrained-imagenet", action="store_true")

    parser.add_argument("--mask-ratio", type=float, default=0.7)
    parser.add_argument("--decoder-dim", type=int, default=192)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--decoder-heads", type=int, default=3)
    parser.add_argument("--mid-focus-prob", type=float, default=0.5)
    parser.add_argument("--mid-band-ratio", type=float, nargs=2, default=[0.2, 0.8])

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--if-valid-set", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu", type=int, default=0)

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_loader, val_loader = make_loaders(args)
    model = BackboneReconstruction(args).to(device)

    ReconstructionTrainer(model, train_loader, val_loader, args, device).train()


if __name__ == "__main__":
    main()
