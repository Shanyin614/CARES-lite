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
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms

from src.config import parse_args
from src.client import FLClient
from src.server import FLServer
from src.model import SmallCNN, TabularMLP

from src.data import (
    set_seed,
    get_device,
    build_client_metas,
    build_dirichlet_client_metas,
    TabularDataset,
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

    elif dataset_name in ["cicids2017", "unsw_nb15"]:
        train_dataset, test_dataset, input_dim, num_classes = load_tabular_dataset(
            args,
            dataset_name,
            data_root,
        )
        input_channels = input_dim
        image_size = 1
    else:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            "Choose from: fashionmnist, fmnist, cifar10, cicids2017, unsw_nb15."
        )

    return train_dataset, test_dataset, input_channels, image_size, num_classes


def _infer_tabular_label_column(dataset_name: str, args) -> str:
    if args.tabular_label_column:
        return args.tabular_label_column

    if dataset_name == "cicids2017":
        return "Label"
    if dataset_name == "unsw_nb15":
        return "attack_cat"

    return "Label"


def _read_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df


def _make_tabular_dataset(
    df: pd.DataFrame,
    label_column: str,
) -> tuple[TabularDataset, np.ndarray, np.ndarray]:
    if label_column not in df.columns:
        raise ValueError(
            f"Label column '{label_column}' not found in tabular dataset. "
            f"Available columns: {list(df.columns)}"
        )

    labels = df[label_column].astype(str).fillna("UNKNOWN").to_numpy(dtype=object)
    features = df.drop(columns=[label_column])
    numeric_features = features.select_dtypes(include=[np.number])

    if numeric_features.shape[1] == 0:
        raise ValueError("No numeric features found in tabular dataset.")

    X = numeric_features.fillna(0.0).to_numpy(dtype=np.float32)
    return X, labels


def load_tabular_dataset(
    args,
    dataset_name: str,
    data_root: str,
):
    label_column = _infer_tabular_label_column(dataset_name, args)

    if args.tabular_train_file and args.tabular_test_file:
        train_df = _read_dataframe(args.tabular_train_file)
        test_df = _read_dataframe(args.tabular_test_file)
    elif args.tabular_all_file:
        all_df = _read_dataframe(args.tabular_all_file)
        train_df, test_df = train_test_split(
            all_df,
            test_size=args.tabular_test_split,
            stratify=all_df[label_column] if label_column in all_df.columns else None,
            random_state=args.seed,
        )
    else:
        default_path = Path(data_root) / (
            "CICIDS2017.csv" if dataset_name == "cicids2017" else "UNSW_NB15.csv"
        )
        if not default_path.exists():
            raise FileNotFoundError(
                f"Expected default tabular dataset at {default_path}. "
                "Please provide --tabular-train-file and --tabular-test-file or --tabular-all-file."
            )
        all_df = _read_dataframe(str(default_path))
        train_df, test_df = train_test_split(
            all_df,
            test_size=args.tabular_test_split,
            stratify=all_df[label_column] if label_column in all_df.columns else None,
            random_state=args.seed,
        )

    X_train, y_train = _make_tabular_dataset(train_df, label_column)
    X_test, y_test = _make_tabular_dataset(test_df, label_column)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    label_values = np.unique(np.concatenate([y_train, y_test]))
    label_map = {val: idx for idx, val in enumerate(sorted(label_values))}

    y_train = np.array([label_map[val] for val in y_train], dtype=np.int64)
    y_test = np.array([label_map[val] for val in y_test], dtype=np.int64)

    train_dataset = TabularDataset(X_train, y_train)
    test_dataset = TabularDataset(X_test, y_test)

    return train_dataset, test_dataset, X_train.shape[1], int(y_train.max() + 1)


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
            num_classes=num_classes,
        )
        for m in metas
    ]

    print(f"\nCreated {len(fl_clients)} FL clients")

    # ── 4. 构造模型工厂，并实例化 FLServer ───────────────
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

        # New profiling-anchor arguments.
        probe_anchor=args.probe_anchor,
        anchor_ema_beta=args.anchor_ema_beta,
        profile_during_training=args.profile_during_training,
    )

    # ── 5. 保存结果 ──────────────────────────────────────
    print("\n" + "=" * 55)
    print(" Final Results")
    print("=" * 55)

    for k, v in metrics.items():
        print(f"  {k}: {v}")

    output_path = Path(args.output_dir) / args.output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "method": "CARES-Lite",
        "dataset": dataset_name,
        "partition": args.partition,
        "num_clients": args.num_clients,
        "num_true_clusters": args.num_true_clusters,
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
        **metrics,
    }

    df = pd.DataFrame([row])
    df.to_csv(output_path, index=False)

    print(f"\nSaved → {output_path}")


if __name__ == "__main__":
    main()
