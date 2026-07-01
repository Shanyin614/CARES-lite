#!/usr/bin/env python
"""CARES-Lite entry point.

This version keeps the original CARES-Lite algorithm path but separates NIDS
loading/preprocessing from image loading. For NIDS experiments, use
src.datasets.nids.load_nids_datasets so that leakage columns are dropped and
preprocessors are fit on train only.
"""

import json
import os
from functools import partial
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import pandas as pd
from torchvision import datasets, transforms

from src.client import FLClient
from src.config import parse_args
from src.data import (
    assert_client_partitions,
    build_client_metas,
    build_dirichlet_client_metas,
    get_device,
    set_seed,
)
from src.datasets.nids import load_nids_datasets
from src.model import SmallCNN, TabularMLP
from src.server import FLServer


def _get_data_root(args) -> str:
    if hasattr(args, "data_dir"):
        return args.data_dir
    if hasattr(args, "data_root"):
        return args.data_root
    return "./data"


def load_image_datasets(args, dataset_name: str, data_root: str):
    """Load Fashion-MNIST or CIFAR-10 with deterministic transforms."""
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
        return train_dataset, test_dataset, 1, 28, 10, {}

    if dataset_name == "cifar10":
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
        return train_dataset, test_dataset, 3, 32, 10, {}

    raise ValueError(f"Unsupported image dataset: {dataset_name}")


def load_datasets(args):
    dataset_name = getattr(args, "dataset", "fashionmnist").lower()
    data_root = _get_data_root(args)

    if dataset_name in ["fashionmnist", "fmnist", "cifar10"]:
        return load_image_datasets(args, dataset_name, data_root)

    if dataset_name in ["cicids2017", "unsw_nb15"]:
        train_dataset, test_dataset, input_dim, num_classes, metadata = load_nids_datasets(
            args,
            dataset_name,
            data_root,
        )
        return train_dataset, test_dataset, input_dim, 1, num_classes, metadata

    raise ValueError(
        f"Unsupported dataset: {dataset_name}. Choose from: "
        "fashionmnist, fmnist, cifar10, cicids2017, unsw_nb15."
    )


def _jsonable_true_groups(true_groups):
    return [[int(x) for x in group] for group in true_groups]


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    train_dataset, test_dataset, input_channels, image_size, num_classes, dataset_meta = load_datasets(args)
    dataset_name = getattr(args, "dataset", "fashionmnist").lower()

    print(f"\nDataset: {dataset_name}")
    print(f" input_channels/input_dim: {input_channels}")
    print(f" image_size: {image_size}")
    print(f" num_classes: {num_classes}")
    print(f" train size: {len(train_dataset)}")
    print(f" test size: {len(test_dataset)}")

    # Build client metadata.
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
            num_clusters=args.num_true_clusters,
        )

    assert_client_partitions(metas, require_disjoint_test=True)
    actual_true_clusters = len(set(int(m.group_id) for m in metas))

    print("\nTrue groups (ground-truth for evaluation only):")
    for gid, labels in enumerate(true_groups):
        print(f" G{gid}: {labels}")
    print(f" requested num_true_clusters: {args.num_true_clusters}")
    print(f" actual_true_clusters: {actual_true_clusters}")

    print("\nClient data split example (client 0):")
    print(f" train: {len(metas[0].train_indices)} samples")
    print(f" val: {len(metas[0].val_indices)} samples")
    print(f" test: {len(metas[0].test_indices)} samples")

    fl_clients = [
        FLClient(
            meta=m,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            num_classes=num_classes,
        )
        for m in metas
    ]
    print(f"\nCreated {len(fl_clients)} FL clients")

    if dataset_name in ["fashionmnist", "fmnist", "cifar10"]:
        model_fn = partial(
            SmallCNN,
            input_channels=input_channels,
            image_size=image_size,
            num_classes=num_classes,
        )
    else:
        model_fn = partial(
            TabularMLP,
            input_dim=input_channels,
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
        probe_anchor=args.probe_anchor,
        anchor_ema_beta=args.anchor_ema_beta,
        profile_during_training=args.profile_during_training,
    )

    print("\n" + "=" * 55)
    print(" Final Results")
    print("=" * 55)
    for k, v in metrics.items():
        print(f" {k}: {v}")

    output_path = Path(args.output_dir) / args.output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "method": "CARES-Lite",
        "dataset": dataset_name,
        "tabular_task": getattr(args, "tabular_task", ""),
        "partition": args.partition,
        "num_clients": args.num_clients,
        "num_true_clusters": args.num_true_clusters,  # kept for backward compatibility
        "num_true_clusters_arg": args.num_true_clusters,
        "actual_true_clusters": actual_true_clusters,
        "true_groups": json.dumps(_jsonable_true_groups(true_groups), ensure_ascii=False),
        "total_rounds": args.total_rounds,
        "warmup_rounds": args.warmup_rounds,
        "cluster_interval": args.cluster_interval,
        "probe_pool_size": args.probe_pool_size,
        "probe_sigma": args.probe_sigma,
        "probe_anchor": args.probe_anchor,
        "anchor_ema_beta": args.anchor_ema_beta,
        "profile_during_training": args.profile_during_training,
        "dpmm_max_components": args.dpmm_max_components,
        "dpmm_prior": args.dpmm_prior,
        "min_cluster_size": args.min_cluster_size,
        "client_frac": args.client_frac,
        "local_epochs": args.local_epochs,
        "lr": args.lr,
        "train_samples_per_client": args.train_samples_per_client,
        "test_samples_per_client": args.test_samples_per_client,
        "val_ratio": args.val_ratio,
        **dataset_meta,
        **metrics,
    }

    df = pd.DataFrame([row])
    df.to_csv(output_path, index=False)
    print(f"\nSaved -> {output_path}")


if __name__ == "__main__":
    main()
