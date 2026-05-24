#!/usr/bin/env python
"""
CARES-Lite: Adaptive Clustered Federated Learning (Plaintext)
==============================================================
Entry point: build data → create clients → create server → run pipeline.
"""

import os
from functools import partial
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import pandas as pd
from torchvision import datasets, transforms

from src.config import parse_args
from src.client import FLClient
from src.server import FLServer
from src.model import SmallCNN

from src.data import (
    set_seed,
    get_device,
    build_client_metas,
    build_dirichlet_client_metas,
)


def _get_data_root(args) -> str:
    """
    Be compatible with both argument names:
      --data-dir
      --data-root
    """
    if hasattr(args, "data_dir"):
        return args.data_dir
    if hasattr(args, "data_root"):
        return args.data_root
    return "./data"


def load_datasets(args):
    """
    Load FashionMNIST or CIFAR-10 and return dataset metadata needed by SmallCNN.

    Note:
    For CIFAR-10, this intentionally uses deterministic transforms only.
    In the current CARES-Lite code, client validation sets are split from the
    training dataset object. RandomCrop / RandomHorizontalFlip would therefore
    also affect loss profiling and make DPMM clustering noisier.
    """
    dataset_name = getattr(args, "dataset", "fashionmnist").lower()
    data_root = _get_data_root(args)

    if dataset_name in ["fashionmnist", "fmnist"]:
        mean = (0.2860,)
        std = (0.3530,)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        train_dataset = datasets.FashionMNIST(
            root=data_root,
            train=True,
            download=True,
            transform=transform,
        )

        test_dataset = datasets.FashionMNIST(
            root=data_root,
            train=False,
            download=True,
            transform=transform,
        )

        input_channels = 1
        image_size = 28
        num_classes = 10

    elif dataset_name == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        train_dataset = datasets.CIFAR10(
            root=data_root,
            train=True,
            download=True,
            transform=transform,
        )

        test_dataset = datasets.CIFAR10(
            root=data_root,
            train=False,
            download=True,
            transform=transform,
        )

        input_channels = 3
        image_size = 32
        num_classes = 10

    else:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            "Choose from: fashionmnist, fmnist, cifar10."
        )

    return train_dataset, test_dataset, input_channels, image_size, num_classes


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    # ── 1. 加载数据集 ────────────────────────────────────
    train_dataset, test_dataset, input_channels, image_size, num_classes = load_datasets(args)

    dataset_name = getattr(args, "dataset", "fashionmnist").lower()
    print(f"\nDataset: {dataset_name}")
    print(f"  input_channels: {input_channels}")
    print(f"  image_size:      {image_size}")
    print(f"  num_classes:     {num_classes}")
    print(f"  train size:      {len(train_dataset)}")
    print(f"  test size:       {len(test_dataset)}")

    # ── 2. 构建 client metadata ──────────────────────────
    if args.partition == "dirichlet":
        metas, true_groups = build_dirichlet_client_metas(
            train_dataset,
            test_dataset,
            num_clients=args.num_clients,
            train_samples=args.train_samples_per_client,
            test_samples=args.test_samples_per_client,
            val_ratio=args.val_ratio,
            seed=args.seed,
            num_clusters=args.num_true_clusters,
            alpha_inter=args.dir_alpha_inter,
            alpha_intra=args.dir_alpha_intra,
        )
    else:
        metas, true_groups = build_client_metas(
            train_dataset,
            test_dataset,
            num_clients=args.num_clients,
            train_samples=args.train_samples_per_client,
            test_samples=args.test_samples_per_client,
            major_ratio=args.major_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )

    print("\nTrue groups (ground-truth, for evaluation only):")
    for gid, labels in enumerate(true_groups):
        print(f"  G{gid}: {labels}")

    print("\nClient data split example (client 0):")
    print(f"  train: {len(metas[0].train_indices)} samples")
    print(f"  val:   {len(metas[0].val_indices)} samples")
    print(f"  test:  {len(metas[0].test_indices)} samples")

    # ── 3. 实例化 FLClient ───────────────────────────────
    fl_clients = [
        FLClient(
            meta=m,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        for m in metas
    ]
    print(f"\nCreated {len(fl_clients)} FL clients")

    # ── 4. 构造模型工厂，并实例化 FLServer ───────────────
    model_fn = partial(
        SmallCNN,
        input_channels=input_channels,
        image_size=image_size,
        num_classes=num_classes,
    )

    server = FLServer(
        clients=fl_clients,
        device=device,
        seed=args.seed,
        model_fn=model_fn,
    )

    metrics = server.run(
        total_rounds=args.total_rounds,
        warmup_rounds=args.warmup_rounds,
        cluster_interval=args.cluster_interval,
        min_cluster_size=args.min_cluster_size,
        probe_pool_size=args.probe_pool_size,
        probe_sigma=args.probe_sigma,
        clip_norm=args.clip_norm,
        noise_sigma=args.noise_sigma,
        dpmm_max_comp=args.dpmm_max_components,
        dpmm_prior=args.dpmm_prior,
        client_frac=args.client_frac,
        local_epochs=args.local_epochs,
        lr=args.lr,
    )

    # ── 5. 保存结果 ──────────────────────────────────────
    print("\n" + "=" * 55)
    print(" Final Results")
    print("=" * 55)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    output_path = Path(args.output_dir) / args.output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([{"method": "CARES-Lite", **metrics}])
    df.to_csv(output_path, index=False)
    print(f"\nSaved → {output_path}")


if __name__ == "__main__":
    main()