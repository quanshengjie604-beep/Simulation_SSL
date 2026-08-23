from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from self_supervised_pretrain.augmentations import RadarAugmentations, ViewConfig
from self_supervised_pretrain.data import make_loaders
from self_supervised_pretrain.models import build_contrastive_model
from self_supervised_pretrain.trainers import ContrastiveTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DopplerWild contrastive pretraining.")
    parser.add_argument("--tracklist-csv", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--ckpt-dir", default="self_supervised_pretrain/checkpoints/contrastive")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--crop-seconds", type=float, default=1.0)
    parser.add_argument("--bins-per-second", type=int, default=90)
    parser.add_argument("--resize-doppler", type=int, default=256)
    parser.add_argument("--train-overlap-ratio", type=float, default=0.5)
    parser.add_argument("--uD-mean", type=float, default=15.589631)
    parser.add_argument("--uD-std", type=float, default=8.797207)
    parser.add_argument("--cache-mode", choices=["none", "lazy", "preload"], default="preload")

    parser.add_argument("--model-name", choices=["mobilenet_v2", "resnet18"], default="mobilenet_v2")
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--stem-channels", type=int, default=32)
    parser.add_argument("--use-radar-stem", action="store_true")
    parser.add_argument("--pretrained-imagenet", action="store_true")
    parser.add_argument("--proj-dim", type=int, default=512)
    parser.add_argument("--proj-hidden-dim", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--label-key", choices=["filename_id", "track_id", "file_name"], default="filename_id")

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-4)
    parser.add_argument("--head-learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-train-steps", type=int, default=0, help="Debug/benchmark only: stop each train epoch after N steps.")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--if-valid-set", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--gpu-list",
        default="",
        help="Comma-separated CUDA device ids for DataParallel, e.g. 0,1,2. Empty uses --gpu only.",
    )

    parser.add_argument("--num-global-views", type=int, default=4)
    parser.add_argument("--num-local-views", type=int, default=4)
    parser.add_argument("--global-crop-min", type=float, default=1.0)
    parser.add_argument("--global-crop-max", type=float, default=1.0)
    parser.add_argument("--local-crop-min", type=float, default=1.0)
    parser.add_argument("--local-crop-max", type=float, default=1.0)
    parser.add_argument("--gaussian-noise-std", type=float, default=0.1)
    parser.add_argument("--flip-time-prob", type=float, default=0.1)
    parser.add_argument("--flip-freq-prob", type=float, default=0.5)
    parser.add_argument("--time-shift-prob", type=float, default=0.5)
    parser.add_argument("--time-shift-max-ratio", type=float, default=0.2)
    parser.add_argument("--time-mask-prob", type=float, default=0.2)
    parser.add_argument("--time-mask-max-ratio", type=float, default=0.05)
    parser.add_argument("--freq-mask-prob", type=float, default=0.2)
    parser.add_argument("--freq-mask-max-ratio", type=float, default=0.05)
    parser.add_argument("--patch-mask-prob", type=float, default=0.2)
    parser.add_argument("--patch-mask-num-min", type=int, default=0)
    parser.add_argument("--patch-mask-num-max", type=int, default=2)
    parser.add_argument("--patch-mask-min-time-ratio", type=float, default=0.02)
    parser.add_argument("--patch-mask-max-time-ratio", type=float, default=0.08)
    parser.add_argument("--patch-mask-min-freq-ratio", type=float, default=0.02)
    parser.add_argument("--patch-mask-max-freq-ratio", type=float, default=0.08)
    parser.add_argument("--interference-mix-prob", type=float, default=0.1)
    parser.add_argument("--interference-alpha-min", type=float, default=0.2)
    parser.add_argument("--interference-alpha-max", type=float, default=0.6)
    parser.add_argument("--local-limb-aug-prob", type=float, default=0.4)
    parser.add_argument("--local-limb-gauss-sigma", type=float, default=3.0)
    parser.add_argument("--local-limb-gauss-alpha", type=float, default=2.0)
    parser.add_argument("--local-limb-mask-percentile", type=float, default=60.0)
    parser.add_argument("--local-limb-mix", type=float, default=0.85)
    parser.add_argument("--local-limb-velnorm-smooth", type=int, default=5)
    parser.add_argument("--enforce-disjoint-global-crops", action=argparse.BooleanOptionalAction, default=True)

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
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)
    gpu_list = [int(value) for value in str(args.gpu_list).split(",") if value.strip()]
    if gpu_list and not torch.cuda.is_available():
        raise RuntimeError("--gpu-list was set, but CUDA is not available.")
    if gpu_list:
        args.gpu = gpu_list[0]
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_loader, val_loader = make_loaders(args)
    augmenter = RadarAugmentations(
        num_global_views=args.num_global_views,
        num_local_views=args.num_local_views,
        global_time_crop=ViewConfig(args.global_crop_min, args.global_crop_max),
        local_time_crop=ViewConfig(args.local_crop_min, args.local_crop_max),
        gaussian_noise_std=args.gaussian_noise_std,
        flip_time_prob=args.flip_time_prob,
        flip_freq_prob=args.flip_freq_prob,
        time_shift_prob=args.time_shift_prob,
        time_shift_max_ratio=args.time_shift_max_ratio,
        time_mask_prob=args.time_mask_prob,
        time_mask_max_ratio=args.time_mask_max_ratio,
        freq_mask_prob=args.freq_mask_prob,
        freq_mask_max_ratio=args.freq_mask_max_ratio,
        patch_mask_prob=args.patch_mask_prob,
        patch_mask_num_min=args.patch_mask_num_min,
        patch_mask_num_max=args.patch_mask_num_max,
        patch_mask_min_time_ratio=args.patch_mask_min_time_ratio,
        patch_mask_max_time_ratio=args.patch_mask_max_time_ratio,
        patch_mask_min_freq_ratio=args.patch_mask_min_freq_ratio,
        patch_mask_max_freq_ratio=args.patch_mask_max_freq_ratio,
        interference_mix_prob=args.interference_mix_prob,
        interference_alpha_min=args.interference_alpha_min,
        interference_alpha_max=args.interference_alpha_max,
        local_limb_aug_prob=args.local_limb_aug_prob,
        local_limb_gauss_sigma=args.local_limb_gauss_sigma,
        local_limb_gauss_alpha=args.local_limb_gauss_alpha,
        local_limb_mask_percentile=args.local_limb_mask_percentile,
        local_limb_mix=args.local_limb_mix,
        local_limb_velnorm_smooth=args.local_limb_velnorm_smooth,
        enforce_disjoint_global_crops=args.enforce_disjoint_global_crops,
    )
    model = build_contrastive_model(args).to(device)
    if gpu_list:
        print(f"Using DataParallel devices: {gpu_list}")
        model = torch.nn.DataParallel(model, device_ids=gpu_list)

    ContrastiveTrainer(model, augmenter, train_loader, val_loader, args, device).train()


if __name__ == "__main__":
    main()
