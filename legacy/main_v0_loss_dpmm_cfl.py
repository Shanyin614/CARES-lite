# legacy/main_v0_loss_dpmm_cfl.py
# -*- coding: utf-8 -*-

"""Legacy monolithic experiment script kept for reference."""

import argparse
import copy
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# -----------------------------
# 0. Utilities
# -----------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ClientData:
    client_id: int
    group_id: int
    train_indices: List[int]
    test_indices: List[int]


# -----------------------------
# 1. Model
# -----------------------------

class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# -----------------------------
# 2. Data partition
# -----------------------------

def label_to_indices(dataset) -> Dict[int, np.ndarray]:
    targets = dataset.targets
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    else:
        targets = np.array(targets)

    pools = {}
    for y in range(10):
        pools[y] = np.where(targets == y)[0]
    return pools


def sample_indices_from_label_mixture(
    label_pools: Dict[int, np.ndarray],
    major_labels: List[int],
    num_samples: int,
    major_ratio: float,
    rng: np.random.Generator,
) -> List[int]:
    all_labels = list(range(10))
    background_labels = [y for y in all_labels if y not in major_labels]

    n_major = int(round(num_samples * major_ratio))
    n_background = num_samples - n_major

    sampled = []

    for _ in range(n_major):
        y = int(rng.choice(major_labels))
        idx = int(rng.choice(label_pools[y]))
        sampled.append(idx)

    for _ in range(n_background):
        y = int(rng.choice(background_labels))
        idx = int(rng.choice(label_pools[y]))
        sampled.append(idx)

    rng.shuffle(sampled)
    return sampled


def build_clients(
    train_dataset,
    test_dataset,
    num_clients: int,
    train_samples_per_client: int,
    test_samples_per_client: int,
    major_ratio: float,
    seed: int,
) -> Tuple[List[ClientData], List[List[int]]]:
    """
    4 true groups:
    G1 = {0,2,6}
    G2 = {1,3}
    G3 = {4,8}
    G4 = {5,7,9}
    """
    true_groups = [
        [0, 2, 6],
        [1, 3],
        [4, 8],
        [5, 7, 9],
    ]

    rng = np.random.default_rng(seed)

    train_pools = label_to_indices(train_dataset)
    test_pools = label_to_indices(test_dataset)

    clients = []
    for cid in range(num_clients):
        gid = cid % len(true_groups)
        major_labels = true_groups[gid]

        train_indices = sample_indices_from_label_mixture(
            train_pools,
            major_labels,
            train_samples_per_client,
            major_ratio,
            rng,
        )
        test_indices = sample_indices_from_label_mixture(
            test_pools,
            major_labels,
            test_samples_per_client,
            major_ratio,
            rng,
        )

        clients.append(
            ClientData(
                client_id=cid,
                group_id=gid,
                train_indices=train_indices,
                test_indices=test_indices,
            )
        )

    return clients, true_groups


# -----------------------------
# 3. Training helpers
# -----------------------------

def train_model_on_dataset(
    model: nn.Module,
    dataset,
    epochs: int,
    batch_size: int,
    lr: float,
    device,
    num_workers: int = 0,
):
    model.to(device)
    model.train()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for _ in range(epochs):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()

    return model


@torch.no_grad()
def average_loss_on_dataset(
    model: nn.Module,
    dataset,
    batch_size: int,
    device,
    num_workers: int = 0,
) -> float:
    model.to(device)
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    total_loss = 0.0
    total_n = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")

        total_loss += float(loss.item())
        total_n += int(y.numel())

    return total_loss / max(total_n, 1)


@torch.no_grad()
def predict_on_dataset(
    model: nn.Module,
    dataset,
    batch_size: int,
    device,
    num_workers: int = 0,
):
    model.to(device)
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    y_true = []
    y_pred = []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        pred = torch.argmax(logits, dim=1).cpu().numpy()

        y_true.extend(y.numpy().tolist())
        y_pred.extend(pred.tolist())

    return np.array(y_true), np.array(y_pred)


def weighted_average_states(
    states: List[Dict[str, torch.Tensor]],
    weights: List[int],
) -> Dict[str, torch.Tensor]:
    total_weight = float(sum(weights))
    avg_state = copy.deepcopy(states[0])

    for key in avg_state.keys():
        avg_state[key] = avg_state[key].float() * (weights[0] / total_weight)

    for state, weight in zip(states[1:], weights[1:]):
        for key in avg_state.keys():
            avg_state[key] += state[key].float() * (weight / total_weight)

    return avg_state


# -----------------------------
# 4. Probe models and loss profiles
# -----------------------------

def build_probe_datasets(
    train_dataset,
    num_probes: int,
    samples_per_probe: int,
    seed: int,
) -> List[Subset]:
    """
    生成 probe datasets。
    前 10 个 probe 偏向单个 class；
    后面的 probe 用 Dirichlet-skewed label distribution。
    """
    rng = np.random.default_rng(seed)
    pools = label_to_indices(train_dataset)

    probe_datasets = []

    for h in range(num_probes):
        indices = []

        if h < 10:
            focus_label = h
            focus_ratio = 0.75

            n_focus = int(round(samples_per_probe * focus_ratio))
            n_rest = samples_per_probe - n_focus

            for _ in range(n_focus):
                indices.append(int(rng.choice(pools[focus_label])))

            other_labels = [y for y in range(10) if y != focus_label]
            for _ in range(n_rest):
                y = int(rng.choice(other_labels))
                indices.append(int(rng.choice(pools[y])))

        else:
            alpha = np.ones(10) * 0.3
            probs = rng.dirichlet(alpha)

            for _ in range(samples_per_probe):
                y = int(rng.choice(np.arange(10), p=probs))
                indices.append(int(rng.choice(pools[y])))

        rng.shuffle(indices)
        probe_datasets.append(Subset(train_dataset, indices))

    return probe_datasets


def train_probe_models(
    train_dataset,
    num_probes: int,
    samples_per_probe: int,
    probe_epochs: int,
    batch_size: int,
    lr: float,
    device,
    seed: int,
    num_workers: int,
) -> List[nn.Module]:
    probe_datasets = build_probe_datasets(
        train_dataset=train_dataset,
        num_probes=num_probes,
        samples_per_probe=samples_per_probe,
        seed=seed,
    )

    probes = []

    for h, probe_dataset in enumerate(probe_datasets):
        print(f"[Probe] Training probe {h + 1}/{num_probes}")

        set_seed(seed + 1000 + h)
        model = SmallCNN()
        model = train_model_on_dataset(
            model=model,
            dataset=probe_dataset,
            epochs=probe_epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            num_workers=num_workers,
        )
        probes.append(copy.deepcopy(model).cpu())

    return probes


def compute_loss_profiles(
    probes: List[nn.Module],
    clients: List[ClientData],
    train_dataset,
    batch_size: int,
    device,
    num_workers: int,
) -> np.ndarray:
    """
    Z[i, h] = client i 在 probe h 上的平均 CE loss。
    """
    num_clients = len(clients)
    num_probes = len(probes)

    Z = np.zeros((num_clients, num_probes), dtype=np.float32)

    for i, client in enumerate(clients):
        client_dataset = Subset(train_dataset, client.train_indices)

        for h, probe in enumerate(probes):
            loss = average_loss_on_dataset(
                model=probe,
                dataset=client_dataset,
                batch_size=batch_size,
                device=device,
                num_workers=num_workers,
            )
            Z[i, h] = loss

        if (i + 1) % 10 == 0 or i == 0:
            print(f"[LossProfile] Computed {i + 1}/{num_clients} clients")

    return Z


def normalize_clip_profiles(
    Z: np.ndarray,
    clip_norm: float,
    noise_sigma: float,
    seed: int,
) -> np.ndarray:
    """
    本地标准化 + L2 clipping + optional Gaussian noise。
    当前 v0 默认 noise_sigma = 0。
    """
    rng = np.random.default_rng(seed)

    R = Z.copy().astype(np.float32)

    row_mean = R.mean(axis=1, keepdims=True)
    row_std = R.std(axis=1, keepdims=True) + 1e-6
    R = (R - row_mean) / row_std

    norms = np.linalg.norm(R, axis=1, keepdims=True) + 1e-12
    scale = np.minimum(1.0, clip_norm / norms)
    R = R * scale

    if noise_sigma > 0:
        R = R + rng.normal(
            loc=0.0,
            scale=noise_sigma * clip_norm,
            size=R.shape,
        ).astype(np.float32)

    return R


# -----------------------------
# 5. DPMM clustering
# -----------------------------

def run_dpmm_clustering(
    profiles: np.ndarray,
    max_components: int,
    weight_concentration_prior: float,
    seed: int,
) -> Tuple[np.ndarray, int]:
    """
    用 BayesianGaussianMixture 近似 DPMM。
    """
    X = StandardScaler().fit_transform(profiles)

    dpmm = BayesianGaussianMixture(
        n_components=max_components,
        covariance_type="diag",
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=weight_concentration_prior,
        max_iter=1000,
        n_init=10,
        random_state=seed,
        init_params="kmeans",
    )

    assignments = dpmm.fit_predict(X)
    k_pred = int(len(np.unique(assignments)))

    return assignments, k_pred


# -----------------------------
# 6. FedAvg
# -----------------------------

def train_fedavg(
    clients: List[ClientData],
    train_dataset,
    rounds: int,
    client_frac: float,
    local_epochs: int,
    batch_size: int,
    lr: float,
    device,
    seed: int,
    num_workers: int,
    verbose_prefix: str = "FedAvg",
) -> nn.Module:
    rng = np.random.default_rng(seed)

    global_model = SmallCNN().to(device)

    num_clients = len(clients)
    clients_per_round = max(1, int(round(client_frac * num_clients)))

    for r in range(rounds):
        selected_ids = rng.choice(
            np.arange(num_clients),
            size=clients_per_round,
            replace=False,
        )

        local_states = []
        local_weights = []

        global_state = copy.deepcopy(global_model.state_dict())

        for local_idx in selected_ids:
            client = clients[int(local_idx)]
            client_dataset = Subset(train_dataset, client.train_indices)

            local_model = SmallCNN().to(device)
            local_model.load_state_dict(global_state)

            train_model_on_dataset(
                model=local_model,
                dataset=client_dataset,
                epochs=local_epochs,
                batch_size=batch_size,
                lr=lr,
                device=device,
                num_workers=num_workers,
            )

            local_states.append(copy.deepcopy(local_model.cpu().state_dict()))
            local_weights.append(len(client.train_indices))

        avg_state = weighted_average_states(local_states, local_weights)
        global_model.load_state_dict(avg_state)
        global_model.to(device)

        print(f"[{verbose_prefix}] Round {r + 1}/{rounds} done")

    return global_model.cpu()


def train_cluster_fedavg(
    clients: List[ClientData],
    train_dataset,
    assignments: np.ndarray,
    rounds: int,
    client_frac: float,
    local_epochs: int,
    batch_size: int,
    lr: float,
    device,
    seed: int,
    num_workers: int,
) -> Dict[int, nn.Module]:
    cluster_models = {}
    unique_clusters = sorted(np.unique(assignments).tolist())

    for c in unique_clusters:
        cluster_client_ids = np.where(assignments == c)[0].tolist()
        cluster_clients = [clients[i] for i in cluster_client_ids]

        print(f"\n[ClusterFedAvg] Training cluster {c}, size={len(cluster_clients)}")

        model = train_fedavg(
            clients=cluster_clients,
            train_dataset=train_dataset,
            rounds=rounds,
            client_frac=client_frac,
            local_epochs=local_epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            seed=seed + 3000 + int(c),
            num_workers=num_workers,
            verbose_prefix=f"Cluster {c}",
        )

        cluster_models[int(c)] = model

    return cluster_models


# -----------------------------
# 7. Evaluation
# -----------------------------

def evaluate_global_model(
    model: nn.Module,
    clients: List[ClientData],
    test_dataset,
    batch_size: int,
    device,
    num_workers: int,
) -> Dict[str, float]:
    client_accs = []
    client_macro_f1s = []

    all_true = []
    all_pred = []

    total_correct = 0
    total_count = 0

    for client in clients:
        client_test_dataset = Subset(test_dataset, client.test_indices)
        y_true, y_pred = predict_on_dataset(
            model=model,
            dataset=client_test_dataset,
            batch_size=batch_size,
            device=device,
            num_workers=num_workers,
        )

        correct = int((y_true == y_pred).sum())
        count = int(len(y_true))

        acc = correct / max(count, 1)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        client_accs.append(acc)
        client_macro_f1s.append(macro_f1)

        all_true.extend(y_true.tolist())
        all_pred.extend(y_pred.tolist())

        total_correct += correct
        total_count += count

    global_macro_f1 = f1_score(
        np.array(all_true),
        np.array(all_pred),
        average="macro",
        labels=list(range(10)),
        zero_division=0,
    )

    return {
        "client_avg_acc": float(np.mean(client_accs)),
        "micro_acc": float(total_correct / max(total_count, 1)),
        "client_avg_macro_f1": float(np.mean(client_macro_f1s)),
        "global_macro_f1": float(global_macro_f1),
    }


def evaluate_cluster_models(
    cluster_models: Dict[int, nn.Module],
    assignments: np.ndarray,
    clients: List[ClientData],
    test_dataset,
    batch_size: int,
    device,
    num_workers: int,
) -> Dict[str, float]:
    client_accs = []
    client_macro_f1s = []

    all_true = []
    all_pred = []

    total_correct = 0
    total_count = 0

    for i, client in enumerate(clients):
        c = int(assignments[i])
        model = cluster_models[c]

        client_test_dataset = Subset(test_dataset, client.test_indices)

        y_true, y_pred = predict_on_dataset(
            model=model,
            dataset=client_test_dataset,
            batch_size=batch_size,
            device=device,
            num_workers=num_workers,
        )

        correct = int((y_true == y_pred).sum())
        count = int(len(y_true))

        acc = correct / max(count, 1)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        client_accs.append(acc)
        client_macro_f1s.append(macro_f1)

        all_true.extend(y_true.tolist())
        all_pred.extend(y_pred.tolist())

        total_correct += correct
        total_count += count

    global_macro_f1 = f1_score(
        np.array(all_true),
        np.array(all_pred),
        average="macro",
        labels=list(range(10)),
        zero_division=0,
    )

    return {
        "client_avg_acc": float(np.mean(client_accs)),
        "micro_acc": float(total_correct / max(total_count, 1)),
        "client_avg_macro_f1": float(np.mean(client_macro_f1s)),
        "global_macro_f1": float(global_macro_f1),
    }


# -----------------------------
# 8. Local baseline
# -----------------------------

def train_and_eval_local(
    clients: List[ClientData],
    train_dataset,
    test_dataset,
    epochs: int,
    batch_size: int,
    lr: float,
    device,
    seed: int,
    num_workers: int,
) -> Dict[str, float]:
    client_accs = []
    client_macro_f1s = []

    all_true = []
    all_pred = []

    total_correct = 0
    total_count = 0

    for i, client in enumerate(clients):
        print(f"[Local] Training client {i + 1}/{len(clients)}")

        set_seed(seed + 5000 + i)

        model = SmallCNN()
        client_train_dataset = Subset(train_dataset, client.train_indices)
        client_test_dataset = Subset(test_dataset, client.test_indices)

        train_model_on_dataset(
            model=model,
            dataset=client_train_dataset,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            num_workers=num_workers,
        )

        y_true, y_pred = predict_on_dataset(
            model=model,
            dataset=client_test_dataset,
            batch_size=batch_size,
            device=device,
            num_workers=num_workers,
        )

        correct = int((y_true == y_pred).sum())
        count = int(len(y_true))

        acc = correct / max(count, 1)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        client_accs.append(acc)
        client_macro_f1s.append(macro_f1)

        all_true.extend(y_true.tolist())
        all_pred.extend(y_pred.tolist())

        total_correct += correct
        total_count += count

    global_macro_f1 = f1_score(
        np.array(all_true),
        np.array(all_pred),
        average="macro",
        labels=list(range(10)),
        zero_division=0,
    )

    return {
        "client_avg_acc": float(np.mean(client_accs)),
        "micro_acc": float(total_correct / max(total_count, 1)),
        "client_avg_macro_f1": float(np.mean(client_macro_f1s)),
        "global_macro_f1": float(global_macro_f1),
    }


# -----------------------------
# 9. IFCA baseline
# -----------------------------

def ifca_choose_cluster(
    models: List[nn.Module],
    client: ClientData,
    train_dataset,
    batch_size: int,
    device,
    num_workers: int,
) -> int:
    client_dataset = Subset(train_dataset, client.train_indices)

    losses = []
    for model in models:
        loss = average_loss_on_dataset(
            model=model,
            dataset=client_dataset,
            batch_size=batch_size,
            device=device,
            num_workers=num_workers,
        )
        losses.append(loss)

    return int(np.argmin(losses))


def train_ifca(
    clients: List[ClientData],
    train_dataset,
    k: int,
    rounds: int,
    client_frac: float,
    local_epochs: int,
    batch_size: int,
    lr: float,
    device,
    seed: int,
    num_workers: int,
) -> List[nn.Module]:
    rng = np.random.default_rng(seed)

    models = []
    for c in range(k):
        set_seed(seed + 7000 + c)
        models.append(SmallCNN().cpu())

    num_clients = len(clients)
    clients_per_round = max(1, int(round(client_frac * num_clients)))

    for r in range(rounds):
        selected_ids = rng.choice(
            np.arange(num_clients),
            size=clients_per_round,
            replace=False,
        )

        cluster_states = {c: [] for c in range(k)}
        cluster_weights = {c: [] for c in range(k)}

        for client_idx in selected_ids:
            client = clients[int(client_idx)]

            chosen_c = ifca_choose_cluster(
                models=models,
                client=client,
                train_dataset=train_dataset,
                batch_size=batch_size,
                device=device,
                num_workers=num_workers,
            )

            local_model = SmallCNN().to(device)
            local_model.load_state_dict(copy.deepcopy(models[chosen_c].state_dict()))

            client_dataset = Subset(train_dataset, client.train_indices)

            train_model_on_dataset(
                model=local_model,
                dataset=client_dataset,
                epochs=local_epochs,
                batch_size=batch_size,
                lr=lr,
                device=device,
                num_workers=num_workers,
            )

            cluster_states[chosen_c].append(copy.deepcopy(local_model.cpu().state_dict()))
            cluster_weights[chosen_c].append(len(client.train_indices))

        for c in range(k):
            if len(cluster_states[c]) > 0:
                avg_state = weighted_average_states(cluster_states[c], cluster_weights[c])
                models[c].load_state_dict(avg_state)

        print(f"[IFCA K={k}] Round {r + 1}/{rounds} done")

    return [m.cpu() for m in models]


def evaluate_ifca_models(
    models: List[nn.Module],
    clients: List[ClientData],
    train_dataset,
    test_dataset,
    batch_size: int,
    device,
    num_workers: int,
) -> Dict[str, float]:
    assignments = []

    for client in clients:
        c = ifca_choose_cluster(
            models=models,
            client=client,
            train_dataset=train_dataset,
            batch_size=batch_size,
            device=device,
            num_workers=num_workers,
        )
        assignments.append(c)

    assignments = np.array(assignments)

    cluster_model_dict = {c: models[c] for c in range(len(models))}

    metrics = evaluate_cluster_models(
        cluster_models=cluster_model_dict,
        assignments=assignments,
        clients=clients,
        test_dataset=test_dataset,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
    )

    metrics["k_pred"] = int(len(np.unique(assignments)))
    return metrics


# -----------------------------
# 10. Main
# -----------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--output", type=str, default="results_v0.csv")

    parser.add_argument("--num-clients", type=int, default=100)
    parser.add_argument("--train-samples-per-client", type=int, default=400)
    parser.add_argument("--test-samples-per-client", type=int, default=100)
    parser.add_argument("--major-ratio", type=float, default=0.85)

    parser.add_argument("--num-probes", type=int, default=16)
    parser.add_argument("--probe-samples", type=int, default=1200)
    parser.add_argument("--probe-epochs", type=int, default=2)
    parser.add_argument("--probe-lr", type=float, default=0.02)

    parser.add_argument("--clip-norm", type=float, default=5.0)
    parser.add_argument("--noise-sigma", type=float, default=0.0)

    parser.add_argument("--dpmm-max-components", type=int, default=10)
    parser.add_argument("--dpmm-prior", type=float, default=0.03)

    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--client-frac", type=float, default=0.2)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.02)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--run-local", action="store_true")
    parser.add_argument("--local-baseline-epochs", type=int, default=5)

    parser.add_argument("--run-ifca", action="store_true")
    parser.add_argument("--ifca-k-list", type=str, default="4")

    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)
    device = get_device()

    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    train_dataset = datasets.FashionMNIST(
        root=args.data_root,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.FashionMNIST(
        root=args.data_root,
        train=False,
        download=True,
        transform=transform,
    )

    clients, true_groups = build_clients(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        num_clients=args.num_clients,
        train_samples_per_client=args.train_samples_per_client,
        test_samples_per_client=args.test_samples_per_client,
        major_ratio=args.major_ratio,
        seed=args.seed,
    )

    print("\nTrue groups:")
    for gid, labels in enumerate(true_groups):
        print(f"  Group {gid}: labels = {labels}")

    results = []

    # -----------------------------
    # FedAvg baseline
    # -----------------------------
    print("\n========== FedAvg baseline ==========")

    fedavg_model = train_fedavg(
        clients=clients,
        train_dataset=train_dataset,
        rounds=args.rounds,
        client_frac=args.client_frac,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        seed=args.seed + 100,
        num_workers=args.num_workers,
        verbose_prefix="FedAvg",
    )

    fedavg_metrics = evaluate_global_model(
        model=fedavg_model,
        clients=clients,
        test_dataset=test_dataset,
        batch_size=args.batch_size,
        device=device,
        num_workers=args.num_workers,
    )

    results.append({
        "method": "FedAvg",
        "k_setting": "-",
        "k_pred": "-",
        **fedavg_metrics,
    })

    print("[FedAvg]", fedavg_metrics)

    # -----------------------------
    # Optional Local baseline
    # -----------------------------
    if args.run_local:
        print("\n========== Local baseline ==========")

        local_metrics = train_and_eval_local(
            clients=clients,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            epochs=args.local_baseline_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            seed=args.seed + 200,
            num_workers=args.num_workers,
        )

        results.append({
            "method": "Local",
            "k_setting": "-",
            "k_pred": "-",
            **local_metrics,
        })

        print("[Local]", local_metrics)

    # -----------------------------
    # Loss-DPMM + Cluster FedAvg
    # -----------------------------
    print("\n========== Loss-DPMM clustering ==========")

    probes = train_probe_models(
        train_dataset=train_dataset,
        num_probes=args.num_probes,
        samples_per_probe=args.probe_samples,
        probe_epochs=args.probe_epochs,
        batch_size=args.batch_size,
        lr=args.probe_lr,
        device=device,
        seed=args.seed + 300,
        num_workers=args.num_workers,
    )

    Z = compute_loss_profiles(
        probes=probes,
        clients=clients,
        train_dataset=train_dataset,
        batch_size=args.batch_size,
        device=device,
        num_workers=args.num_workers,
    )

    profiles = normalize_clip_profiles(
        Z=Z,
        clip_norm=args.clip_norm,
        noise_sigma=args.noise_sigma,
        seed=args.seed + 400,
    )

    assignments, k_pred = run_dpmm_clustering(
        profiles=profiles,
        max_components=args.dpmm_max_components,
        weight_concentration_prior=args.dpmm_prior,
        seed=args.seed + 500,
    )

    print(f"\n[Loss-DPMM] K_pred = {k_pred}")
    unique, counts = np.unique(assignments, return_counts=True)
    for c, n in zip(unique, counts):
        print(f"  cluster {int(c)}: {int(n)} clients")

    print("\n========== Cluster FedAvg after Loss-DPMM ==========")

    cluster_models = train_cluster_fedavg(
        clients=clients,
        train_dataset=train_dataset,
        assignments=assignments,
        rounds=args.rounds,
        client_frac=args.client_frac,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        seed=args.seed + 600,
        num_workers=args.num_workers,
    )

    loss_dpmm_metrics = evaluate_cluster_models(
        cluster_models=cluster_models,
        assignments=assignments,
        clients=clients,
        test_dataset=test_dataset,
        batch_size=args.batch_size,
        device=device,
        num_workers=args.num_workers,
    )

    results.append({
        "method": "Loss-DPMM + ClusterFedAvg",
        "k_setting": "auto",
        "k_pred": k_pred,
        **loss_dpmm_metrics,
    })

    print("[Loss-DPMM + ClusterFedAvg]", loss_dpmm_metrics)

    # -----------------------------
    # Optional IFCA baseline
    # -----------------------------
    if args.run_ifca:
        print("\n========== IFCA baseline ==========")

        ifca_k_list = [int(x.strip()) for x in args.ifca_k_list.split(",") if x.strip()]

        for k in ifca_k_list:
            print(f"\n========== IFCA K={k} ==========")

            ifca_models = train_ifca(
                clients=clients,
                train_dataset=train_dataset,
                k=k,
                rounds=args.rounds,
                client_frac=args.client_frac,
                local_epochs=args.local_epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=device,
                seed=args.seed + 800 + k,
                num_workers=args.num_workers,
            )

            ifca_metrics = evaluate_ifca_models(
                models=ifca_models,
                clients=clients,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                batch_size=args.batch_size,
                device=device,
                num_workers=args.num_workers,
            )

            results.append({
                "method": "IFCA",
                "k_setting": k,
                "k_pred": ifca_metrics.pop("k_pred"),
                **ifca_metrics,
            })

            print(f"[IFCA K={k}]", ifca_metrics)

    # -----------------------------
    # Save results
    # -----------------------------
    df = pd.DataFrame(results)

    metric_cols = [
        "client_avg_acc",
        "micro_acc",
        "client_avg_macro_f1",
        "global_macro_f1",
    ]

    for col in metric_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)

    print("\n========== Final results ==========")
    print(df)

    df.to_csv(args.output, index=False)
    print(f"\nSaved results to: {args.output}")


if __name__ == "__main__":
    main()
