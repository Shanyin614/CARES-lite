#!/usr/bin/env python
"""
CARES-Lite: Adaptive Clustered Federated Learning (Plaintext)
==============================================================
Entry point: build data → create clients → create server → run pipeline.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
from pathlib import Path

import pandas as pd
from torchvision import datasets, transforms

from src.config import parse_args

from src.client import FLClient
from src.server import FLServer

from src.data import (
    set_seed,
    get_device,
    build_client_metas,
    build_dirichlet_client_metas,
)

def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    # ── 1. 加载数据集 ────────────────────────────────────
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    train_dataset = datasets.FashionMNIST(
        root=args.data_root, train=True,
        download=True, transform=transform,
    )
    test_dataset = datasets.FashionMNIST(
        root=args.data_root, train=False,
        download=True, transform=transform,
    )

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

    print(f"\nClient data split example (client 0):")
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

    # ── 4. 实例化 FLServer 并运行 pipeline ───────────────
    server = FLServer(
        clients=fl_clients,
        device=device,
        seed=args.seed,
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
    df = pd.DataFrame([{"method": "CARES-Lite", **metrics}])
    df.to_csv(output_path, index=False)
    print(f"\nSaved → {output_path}")


if __name__ == "__main__":
    main()
