"""CARES-Lite experiment configuration."""

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
DATA_DIR = PROJECT_ROOT / "data"


def parse_args():
    p = argparse.ArgumentParser(
        description="CARES-Lite: Adaptive Clustered Federated Learning"
    )

    # General
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", type=str, default=str(DATA_DIR))
    p.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    p.add_argument("--output-name", type=str, default="results.csv")

    # Dataset
    p.add_argument(
        "--dataset",
        type=str,
        default="fashionmnist",
        choices=["fashionmnist", "fmnist", "cifar10", "cicids2017", "unsw_nb15"],
        help="Dataset used in the experiment",
    )
    p.add_argument(
        "--tabular-task",
        type=str,
        default="binary",
        choices=["binary", "multiclass"],
        help=(
            "Task definition for tabular NIDS datasets. "
            "UNSW binary uses label as target and drops attack_cat; "
            "UNSW multiclass uses attack_cat as target and drops label."
        ),
    )
    p.add_argument(
        "--tabular-train-file",
        type=str,
        default="",
        help="Path to a tabular training CSV file for tabular datasets",
    )
    p.add_argument(
        "--tabular-test-file",
        type=str,
        default="",
        help="Path to a tabular test CSV file for tabular datasets",
    )
    p.add_argument(
        "--tabular-all-file",
        type=str,
        default="",
        help=(
            "Path to a single tabular CSV file containing all examples. "
            "This is for debugging only; prefer official train/test splits for NIDS."
        ),
    )
    p.add_argument(
        "--tabular-label-column",
        type=str,
        default="",
        help="Label column name for tabular datasets; default depends on dataset/task",
    )
    p.add_argument(
        "--tabular-test-split",
        type=float,
        default=0.2,
        help="Test split fraction when using a single tabular CSV file",
    )
    p.add_argument(
        "--tabular-drop-columns",
        nargs="*",
        default=[],
        help="Extra feature columns to drop from tabular datasets",
    )

    # Client data partition
    p.add_argument("--num-clients", type=int, default=100)
    p.add_argument("--train-samples-per-client", type=int, default=400)
    p.add_argument("--test-samples-per-client", type=int, default=100)
    p.add_argument("--major-ratio", type=float, default=0.85)
    p.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of client train data held out as local validation",
    )
    p.add_argument(
        "--partition",
        type=str,
        default="manual",
        choices=["manual", "dirichlet"],
        help="Client data partition type",
    )
    p.add_argument(
        "--num-true-clusters",
        type=int,
        default=10,
        help=(
            "Requested latent ground-truth groups. Used directly for Dirichlet; "
            "manual partition may have fewer actual groups depending on labels."
        ),
    )
    p.add_argument(
        "--dir-alpha-inter",
        type=float,
        default=0.1,
        help="Dirichlet alpha for inter-cluster label distributions",
    )
    p.add_argument(
        "--dir-alpha-intra",
        type=float,
        default=10.0,
        help="Dirichlet alpha for intra-cluster client distributions",
    )

    # Probe pool / profiling
    p.add_argument(
        "--probe-pool-size",
        type=int,
        default=16,
        help="M: probe count, including original, class-ablation, and random probes",
    )
    p.add_argument(
        "--probe-sigma",
        type=float,
        default=0.05,
        help="Random perturbation probe noise standard deviation",
    )
    p.add_argument(
        "--probe-anchor",
        type=str,
        default="ema_global",
        choices=["assigned", "global", "ema_global"],
        help=(
            "Base model used for dynamic re-clustering profiling. "
            "assigned = old behavior; global = weighted average of group models; "
            "ema_global = EMA-smoothed global anchor."
        ),
    )
    p.add_argument(
        "--anchor-ema-beta",
        type=float,
        default=0.9,
        help="EMA beta for the global profiling anchor",
    )
    p.add_argument(
        "--profile-during-training",
        action="store_true",
        help=(
            "If set, compute loss profiles during ordinary training rounds. "
            "Default is off; profiles are computed only during full profiling / re-clustering."
        ),
    )

    # Loss profile normalization / DP
    p.add_argument("--clip-norm", type=float, default=5.0)
    p.add_argument(
        "--noise-sigma",
        type=float,
        default=0.0,
        help="DP noise sigma; 0 means no noise",
    )

    # DPMM clustering
    p.add_argument(
        "--dpmm-max-components",
        type=int,
        default=10,
        help="K_max: truncated DPMM upper bound",
    )
    p.add_argument(
        "--dpmm-prior",
        type=float,
        default=0.03,
        help="Dirichlet process concentration prior",
    )
    p.add_argument(
        "--min-cluster-size",
        type=int,
        default=5,
        help="T_min: minimum group size; smaller groups get merged",
    )

    # Federated training
    p.add_argument(
        "--total-rounds",
        type=int,
        default=30,
        help="Total communication rounds, including warmup and clustered training",
    )
    p.add_argument(
        "--warmup-rounds",
        type=int,
        default=5,
        help="T_warm: global FedAvg rounds before first clustering",
    )
    p.add_argument(
        "--cluster-interval",
        type=int,
        default=10,
        help="Re-run DPMM every tau clustered rounds after warmup",
    )
    p.add_argument("--client-frac", type=float, default=0.2)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.02)

    # Misc
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)

    args = p.parse_args()

    if args.cluster_interval <= 0:
        raise ValueError("--cluster-interval must be positive")
    if args.probe_pool_size < 1:
        raise ValueError("--probe-pool-size must be at least 1")
    if not 0.0 <= args.anchor_ema_beta <= 1.0:
        raise ValueError("--anchor-ema-beta must be in [0, 1]")
    if not 0.0 < args.tabular_test_split < 1.0:
        raise ValueError("--tabular-test-split must be in (0, 1)")
    if not 0.0 <= args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be in [0, 1)")
    if args.dataset == "unsw_nb15" and args.tabular_task not in {"binary", "multiclass"}:
        raise ValueError("UNSW_NB15 requires --tabular-task binary or multiclass")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    return args
