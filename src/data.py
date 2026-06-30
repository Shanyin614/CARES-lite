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


class TabularDataset(Dataset):
    """Simple tabular dataset wrapper for NumPy features and integer labels."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = features.astype(np.float32)
        self.targets = labels.astype(np.int64)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return torch.from_numpy(self.features[idx]), int(self.targets[idx])


# ╔══════════════════════════════════════════════════════════════════════════════════╗
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
    labels = np.unique(targets)
    return {int(y): np.where(targets == y)[0] for y in labels}


def _sample_mixture(
    pools: Dict[int, np.ndarray],
    major_labels: List[int],
    n: int,
    major_ratio: float,
    rng: np.random.Generator,
) -> List[int]:
    major_labels = [y for y in major_labels if y in pools]
    bg = [y for y in sorted(pools.keys()) if y not in major_labels]
    n_major = int(round(n * major_ratio))

    sampled: List[int] = []
    for _ in range(n_major):
        if len(major_labels) == 0:
            raise ValueError("No major labels available for _sample_mixture")
        y = int(rng.choice(major_labels))
        sampled.append(int(rng.choice(pools[y])))
    for _ in range(n - n_major):
        if len(bg) == 0:
            raise ValueError("No background labels available for _sample_mixture")
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

    labels = sorted(train_pools.keys())
    if all(y in train_pools for group in TRUE_GROUPS for y in group):
        true_groups = TRUE_GROUPS
    else:
        num_groups = min(len(TRUE_GROUPS), len(labels))
        if num_groups == 0:
            raise ValueError("No labels found in training dataset for manual partitioning")
        true_groups = [labels[i::num_groups] for i in range(num_groups)]

    metas: List[ClientMeta] = []
    for cid in range(num_clients):
        gid = cid % len(true_groups)

        # 先采样完整的训练索引
        major_labels = [y for y in true_groups[gid] if y in train_pools]
        all_train = _sample_mixture(
            train_pools, major_labels, train_samples, major_ratio, rng,
        )

        # 拆分 train / val
        n_val = max(1, int(round(len(all_train) * val_ratio)))
        val_indices = all_train[:n_val]
        train_indices = all_train[n_val:]

        test_indices = _sample_mixture(
            test_pools, major_labels, test_samples, major_ratio, rng,
        )

        metas.append(ClientMeta(
            client_id=cid,
            group_id=gid,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
        ))

    return metas, true_groups
def _sample_by_label_probs(
    pools: Dict[int, np.ndarray],
    probs: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> List[int]:
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()

    sampled: List[int] = []
    labels = np.array(sorted(pools.keys()), dtype=int)
    selected = rng.choice(labels, size=n, p=probs)

    for y in selected:
        sampled.append(int(rng.choice(pools[int(y)])))

    rng.shuffle(sampled)
    return sampled


def build_dirichlet_client_metas(
    train_dataset: Dataset,
    test_dataset: Dataset,
    num_clients: int,
    train_samples: int,
    test_samples: int,
    val_ratio: float,
    seed: int,
    num_clusters: int = 10,
    alpha_inter: float = 0.1,
    alpha_intra: float = 10.0,
) -> Tuple[List[ClientMeta], List[List[int]]]:
    """
    DPMM-CFL-style two-level Dirichlet partition.

    Level 1:
        Each ground-truth cluster k has a label distribution pi_k
        drawn from Dirichlet(alpha_inter).

    Level 2:
        Each client inside cluster k has a local label distribution
        theta_i drawn around pi_k using Dirichlet(alpha_intra * pi_k).

    group_id is the ground-truth cluster id, used only for ARI/NMI/Purity.
    """
    rng = np.random.default_rng(seed)
    train_pools = label_to_indices(train_dataset)
    test_pools = label_to_indices(test_dataset)

    labels = np.array(sorted(train_pools.keys()), dtype=int)
    num_labels = len(labels)

    # K ground-truth cluster-level label distributions
    cluster_priors = rng.dirichlet(
        alpha=np.full(num_labels, alpha_inter, dtype=np.float64),
        size=num_clusters,
    )

    metas: List[ClientMeta] = []

    for cid in range(num_clients):
        gid = cid % num_clusters
        pi_k = cluster_priors[gid]

        # Client distribution around its cluster distribution
        theta_i = rng.dirichlet(alpha_intra * pi_k + 1e-6)

        all_train = _sample_by_label_probs(
            train_pools, theta_i, train_samples, rng
        )

        n_val = max(1, int(round(len(all_train) * val_ratio)))
        val_indices = all_train[:n_val]
        train_indices = all_train[n_val:]

        test_indices = _sample_by_label_probs(
            test_pools, theta_i, test_samples, rng
        )

        metas.append(ClientMeta(
            client_id=cid,
            group_id=gid,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
        ))

    # 只是为了打印展示：每个真实簇 top-3 dominant labels
    true_groups = [
        [int(labels[x]) for x in np.argsort(-cluster_priors[k])[:3]]
        for k in range(num_clusters)
    ]

    return metas, true_groups
