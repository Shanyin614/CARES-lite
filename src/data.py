"""Data partitioning, client metadata, and shared utilities."""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# ╔══════════════════════════════════════════════════════════╗
# ║  Utilities                                               ║
# ╚══════════════════════════════════════════════════════════╝

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ╔══════════════════════════════════════════════════════════╗
# ║  Client metadata                                         ║
# ╚══════════════════════════════════════════════════════════╝

@dataclass
class ClientMeta:
    """Lightweight metadata describing one client's data partition."""
    client_id: int
    group_id: int                                     # ground-truth (evaluation only)
    train_indices: List[int] = field(repr=False)      # 用于本地训练
    val_indices: List[int] = field(repr=False)        # 用于计算 loss profile
    test_indices: List[int] = field(repr=False)       # 用于评估


# ╔══════════════════════════════════════════════════════════╗
# ║  Ground-truth group definition                           ║
# ╚══════════════════════════════════════════════════════════╝

TRUE_GROUPS: List[List[int]] = [
    [0, 2, 6],   # T-shirt, Pullover, Shirt
    [1, 3],       # Trouser, Dress
    [4, 8],       # Coat, Bag
    [5, 7, 9],   # Sandal, Sneaker, Ankle boot
]


# ╔══════════════════════════════════════════════════════════╗
# ║  Helpers                                                 ║
# ╚══════════════════════════════════════════════════════════╝

def label_to_indices(dataset: Dataset) -> Dict[int, np.ndarray]:
    targets = dataset.targets
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    else:
        targets = np.array(targets)
    return {y: np.where(targets == y)[0] for y in range(10)}


def _sample_mixture(
    pools: Dict[int, np.ndarray],
    major_labels: List[int],
    n: int,
    major_ratio: float,
    rng: np.random.Generator,
) -> List[int]:
    bg = [y for y in range(10) if y not in major_labels]
    n_major = int(round(n * major_ratio))

    sampled: List[int] = []
    for _ in range(n_major):
        y = int(rng.choice(major_labels))
        sampled.append(int(rng.choice(pools[y])))
    for _ in range(n - n_major):
        y = int(rng.choice(bg))
        sampled.append(int(rng.choice(pools[y])))

    rng.shuffle(sampled)
    return sampled


# ╔══════════════════════════════════════════════════════════╗
# ║  Build client partitions (with train / val / test split) ║
# ╚══════════════════════════════════════════════════════════╝

def build_client_metas(
    train_dataset: Dataset,
    test_dataset: Dataset,
    num_clients: int,
    train_samples: int,
    test_samples: int,
    major_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[ClientMeta], List[List[int]]]:
    """
    为每个 client 分配 Non-IID 数据，
    并将 train 数据按 val_ratio 拆分为 train + val。
    """
    rng = np.random.default_rng(seed)
    train_pools = label_to_indices(train_dataset)
    test_pools = label_to_indices(test_dataset)

    metas: List[ClientMeta] = []
    for cid in range(num_clients):
        gid = cid % len(TRUE_GROUPS)

        # 先采样完整的训练索引
        all_train = _sample_mixture(
            train_pools, TRUE_GROUPS[gid], train_samples, major_ratio, rng,
        )

        # 拆分 train / val
        n_val = max(1, int(round(len(all_train) * val_ratio)))
        val_indices = all_train[:n_val]
        train_indices = all_train[n_val:]

        test_indices = _sample_mixture(
            test_pools, TRUE_GROUPS[gid], test_samples, major_ratio, rng,
        )

        metas.append(ClientMeta(
            client_id=cid,
            group_id=gid,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
        ))

    return metas, TRUE_GROUPS
