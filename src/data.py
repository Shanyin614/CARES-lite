"""Data partitioning, client metadata, and shared utilities.

This version keeps the original CARES-Lite public interfaces but fixes the
most important experimental issue for NIDS: client samples are assigned
without replacement, so one raw example is not duplicated across clients.
"""

import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# Utilities

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


@dataclass
class ClientMeta:
    """Lightweight metadata describing one client's data partition."""

    client_id: int
    group_id: int  # ground truth for evaluation only
    train_indices: List[int] = field(repr=False)
    val_indices: List[int] = field(repr=False)
    test_indices: List[int] = field(repr=False)


# Original Fashion-MNIST label groups. Use only when labels are exactly compatible.
TRUE_GROUPS: List[List[int]] = [
    [0, 2, 6],
    [1, 3],
    [4, 8],
    [5, 7, 9],
]


def _as_numpy_targets(dataset: Dataset) -> np.ndarray:
    targets = getattr(dataset, "targets")
    if isinstance(targets, torch.Tensor):
        return targets.cpu().numpy()
    return np.asarray(targets)


def label_to_indices(dataset: Dataset) -> Dict[int, np.ndarray]:
    targets = _as_numpy_targets(dataset)
    labels = np.unique(targets)
    return {int(y): np.where(targets == y)[0].astype(int) for y in labels}


class _NoReplacementLabelAllocator:
    """Label-aware index allocator that never returns the same index twice."""

    def __init__(self, pools: Dict[int, np.ndarray], rng: np.random.Generator):
        self.rng = rng
        self.pools: Dict[int, List[int]] = {}
        for y, indices in pools.items():
            shuffled = rng.permutation(np.asarray(indices, dtype=int)).tolist()
            self.pools[int(y)] = shuffled

    def remaining_labels(self) -> List[int]:
        return [y for y, idxs in self.pools.items() if len(idxs) > 0]

    def remaining_count(self) -> int:
        return sum(len(idxs) for idxs in self.pools.values())

    def take_by_probs(self, probs: Dict[int, float], n: int) -> List[int]:
        """Take up to n samples without replacement according to label probabilities.

        If requested labels are exhausted, the allocator falls back to all labels
        that still have remaining examples. This avoids silently sampling with
        replacement while making small/imbalanced datasets usable.
        """
        out: List[int] = []
        for _ in range(n):
            available = [y for y in self.remaining_labels() if probs.get(y, 0.0) > 0]
            if not available:
                available = self.remaining_labels()
            if not available:
                break
            weights = np.array([max(float(probs.get(y, 0.0)), 0.0) for y in available])
            if weights.sum() <= 0:
                weights = np.ones(len(available), dtype=np.float64)
            weights = weights / weights.sum()
            y = int(self.rng.choice(np.array(available, dtype=int), p=weights))
            out.append(int(self.pools[y].pop()))
        self.rng.shuffle(out)
        return out


def _split_train_val(
    indices: List[int], val_ratio: float, rng: np.random.Generator
) -> Tuple[List[int], List[int]]:
    indices = list(indices)
    rng.shuffle(indices)
    if len(indices) <= 1 or val_ratio <= 0:
        return indices, []
    n_val = int(round(len(indices) * val_ratio))
    n_val = min(max(1, n_val), len(indices) - 1)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    return train_indices, val_indices


def _make_manual_groups(labels: Iterable[int], num_groups: int | None = None) -> List[List[int]]:
    labels = sorted(int(y) for y in labels)
    label_set = set(labels)

    # Preserve the original Fashion-MNIST controlled split when exactly applicable.
    if all(set(g).issubset(label_set) for g in TRUE_GROUPS) and len(label_set) >= 10:
        if num_groups in (None, 0, len(TRUE_GROUPS)):
            return [list(g) for g in TRUE_GROUPS]

    if num_groups is None or num_groups <= 0:
        num_groups = min(len(TRUE_GROUPS), len(labels))

    # Manual label groups cannot exceed the number of labels unless we introduce
    # artificial distributional groups. Use Dirichlet partition for that purpose.
    num_groups = max(1, min(int(num_groups), len(labels)))
    return [labels[i::num_groups] for i in range(num_groups)]


def _manual_label_probs(
    all_labels: List[int], major_labels: List[int], major_ratio: float
) -> Dict[int, float]:
    major_labels = [int(y) for y in major_labels if y in all_labels]
    bg_labels = [int(y) for y in all_labels if y not in major_labels]
    probs = {int(y): 0.0 for y in all_labels}

    if not major_labels:
        for y in all_labels:
            probs[int(y)] = 1.0 / len(all_labels)
        return probs

    if not bg_labels:
        for y in major_labels:
            probs[int(y)] = 1.0 / len(major_labels)
        return probs

    for y in major_labels:
        probs[int(y)] = float(major_ratio) / len(major_labels)
    for y in bg_labels:
        probs[int(y)] = float(1.0 - major_ratio) / len(bg_labels)
    return probs


def build_client_metas(
    train_dataset: Dataset,
    test_dataset: Dataset,
    num_clients: int,
    train_samples: int,
    test_samples: int,
    major_ratio: float,
    val_ratio: float,
    seed: int,
    num_clusters: int | None = None,
) -> Tuple[List[ClientMeta], List[List[int]]]:
    """Build manual label-skew client partitions without replacement."""
    rng = np.random.default_rng(seed)
    train_pools = label_to_indices(train_dataset)
    test_pools = label_to_indices(test_dataset)
    labels = sorted(train_pools.keys())

    if len(labels) == 0:
        raise ValueError("No labels found in training dataset for manual partitioning")

    true_groups = _make_manual_groups(labels, num_groups=num_clusters)
    train_alloc = _NoReplacementLabelAllocator(train_pools, rng)
    test_alloc = _NoReplacementLabelAllocator(test_pools, rng)

    metas: List[ClientMeta] = []
    for cid in range(num_clients):
        gid = cid % len(true_groups)
        probs = _manual_label_probs(labels, true_groups[gid], major_ratio)

        client_train_all = train_alloc.take_by_probs(probs, train_samples)
        train_indices, val_indices = _split_train_val(client_train_all, val_ratio, rng)
        test_indices = test_alloc.take_by_probs(probs, test_samples)

        if len(train_indices) == 0:
            raise RuntimeError(
                f"Client {cid} received no training samples. "
                "Reduce --num-clients or --train-samples-per-client."
            )

        metas.append(
            ClientMeta(
                client_id=cid,
                group_id=gid,
                train_indices=train_indices,
                val_indices=val_indices,
                test_indices=test_indices,
            )
        )

    assert_client_partitions(metas, require_disjoint_test=True)
    return metas, true_groups


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
    """Two-level Dirichlet client partitions without replacement.

    Level 1: each latent group k has a label distribution pi_k.
    Level 2: each client in group k has a local distribution theta_i around pi_k.
    group_id is used only for ARI/NMI/Purity evaluation.
    """
    if num_clusters <= 0:
        raise ValueError("num_clusters must be positive for Dirichlet partition")

    rng = np.random.default_rng(seed)
    train_pools = label_to_indices(train_dataset)
    test_pools = label_to_indices(test_dataset)
    labels = np.array(sorted(train_pools.keys()), dtype=int)
    if len(labels) == 0:
        raise ValueError("No labels found in training dataset for Dirichlet partitioning")

    num_labels = len(labels)
    cluster_priors = rng.dirichlet(
        alpha=np.full(num_labels, alpha_inter, dtype=np.float64),
        size=num_clusters,
    )

    train_alloc = _NoReplacementLabelAllocator(train_pools, rng)
    test_alloc = _NoReplacementLabelAllocator(test_pools, rng)

    metas: List[ClientMeta] = []
    for cid in range(num_clients):
        gid = cid % num_clusters
        pi_k = cluster_priors[gid]
        theta_i = rng.dirichlet(alpha_intra * pi_k + 1e-6)
        probs = {int(y): float(p) for y, p in zip(labels.tolist(), theta_i.tolist())}

        client_train_all = train_alloc.take_by_probs(probs, train_samples)
        train_indices, val_indices = _split_train_val(client_train_all, val_ratio, rng)
        test_indices = test_alloc.take_by_probs(probs, test_samples)

        if len(train_indices) == 0:
            raise RuntimeError(
                f"Client {cid} received no training samples. "
                "Reduce --num-clients or --train-samples-per-client."
            )

        metas.append(
            ClientMeta(
                client_id=cid,
                group_id=gid,
                train_indices=train_indices,
                val_indices=val_indices,
                test_indices=test_indices,
            )
        )

    true_groups = [
        [int(labels[x]) for x in np.argsort(-cluster_priors[k])[: min(3, num_labels)]]
        for k in range(num_clusters)
    ]

    assert_client_partitions(metas, require_disjoint_test=True)
    return metas, true_groups


def assert_client_partitions(
    metas: List[ClientMeta], require_disjoint_test: bool = True
) -> None:
    """Fail fast when a split contains duplicated raw examples."""
    all_train_val: List[int] = []
    all_test: List[int] = []

    for meta in metas:
        train_set = set(meta.train_indices)
        val_set = set(meta.val_indices)
        if train_set.intersection(val_set):
            raise AssertionError(f"Client {meta.client_id} has train/val overlap")
        all_train_val.extend(meta.train_indices)
        all_train_val.extend(meta.val_indices)
        all_test.extend(meta.test_indices)

    if len(all_train_val) != len(set(all_train_val)):
        dup = len(all_train_val) - len(set(all_train_val))
        raise AssertionError(f"Train/val indices overlap across clients: {dup} duplicates")

    if require_disjoint_test and len(all_test) != len(set(all_test)):
        dup = len(all_test) - len(set(all_test))
        raise AssertionError(f"Test indices overlap across clients: {dup} duplicates")