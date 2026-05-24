"""CARES-Lite experiment configuration."""
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

    # ── General ───────────────────────────────
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", type=str, default=str(DATA_DIR))
    p.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    p.add_argument("--output-name", type=str, default="results.csv")

    # ── Client data partition ─────────────────
    p.add_argument("--num-clients", type=int, default=100)
    p.add_argument("--train-samples-per-client", type=int, default=400)
    p.add_argument("--test-samples-per-client", type=int, default=100)
    p.add_argument("--major-ratio", type=float, default=0.85)
    p.add_argument("--val-ratio", type=float, default=0.2,
                   help="Fraction of client train data held out as local validation")

    # ── Probe pool (client-side) ──────────────
    p.add_argument("--probe-pool-size", type=int, default=16,
                   help="M: probe 数量 (1 original + 10 class-ablation + M-11 random)")
    p.add_argument("--probe-sigma", type=float, default=0.5,
                   help="σ: random perturbation probes 的噪声标准差 (仅影响 probe 11+)")

    # ── Loss profile normalization / DP ───────
    p.add_argument("--clip-norm", type=float, default=5.0)
    p.add_argument("--noise-sigma", type=float, default=0.0,
                   help="DP noise σ (0 = no noise)")

    # ── DPMM clustering ──────────────────────
    p.add_argument("--dpmm-max-components", type=int, default=10,
                   help="K_max: truncated DPMM upper bound")
    p.add_argument("--dpmm-prior", type=float, default=0.03,
                   help="Dirichlet process concentration prior")

    # ── Federated training ────────────────────
    p.add_argument("--total-rounds", type=int, default=30,
                   help="Total communication rounds (warmup + clustered)")
    p.add_argument("--warmup-rounds", type=int, default=5,
                   help="T_warm: rounds of global FedAvg before first clustering")
    p.add_argument("--cluster-interval", type=int, default=5,
                   help="τ: re-run DPMM every τ rounds after warmup")
    p.add_argument("--min-cluster-size", type=int, default=5,
                   help="T_min: minimum group size; smaller groups get merged")
    p.add_argument("--client-frac", type=float, default=0.2)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.02)

    # ── Misc ──────────────────────────────────
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--partition", type=str, default="manual",
               choices=["manual", "dirichlet"])

    p.add_argument("--num-true-clusters", type=int, default=10)
    p.add_argument("--dir-alpha-inter", type=float, default=0.1,
                help="Dirichlet alpha for inter-cluster label distributions")
    p.add_argument("--dir-alpha-intra", type=float, default=10.0,
                help="Dirichlet alpha for intra-cluster client distributions")
    p.add_argument("--dataset",type=str,default="fashionmnist",choices=["fashionmnist", "fmnist", "cifar10"],)

    args = p.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    return args
