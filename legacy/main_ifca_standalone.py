# legacy/main_ifca_standalone.py
# -*- coding: utf-8 -*-

"""Legacy standalone IFCA script kept for reference."""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import copy
import random
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from sklearn.metrics import f1_score


# -----------------------------
# 0. Utilities
# -----------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str):
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
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
):
    """
    4 true groups:
    G0 = {0, 2, 6}
    G1 = {1, 3}
    G2 = {4, 8}
    G3 = {5, 7, 9}
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
            label_pools=train_pools,
            major_labels=major_labels,
            num_samples=train_samples_per_client,
            major_ratio=major_ratio,
            rng=rng,
        )

        test_indices = sample_indices_from_label_mixture(
            label_pools=test_pools,
            major_labels=major_labels,
            num_samples=test_samples_per_client,
            major_ratio=major_ratio,
            rng=rng,
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
# 3. Basic training helpers
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
# 4. IFCA
# -----------------------------

def ifca_choose_cluster(
    models: List[nn.Module],
    client: ClientData,
    train_dataset,
    batch_size: int,
    device,
    num_workers: int,
) -> int:
    """
    IFCA assignment step:
    client chooses the model with the lowest local empirical loss.
    """
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


def assign_all_clients(
    models: List[nn.Module],
    clients: List[ClientData],
    train_dataset,
    batch_size: int,
    device,
    num_workers: int,
) -> np.ndarray:
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

    return np.array(assignments, dtype=np.int64)


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
    """
    IFCA training.

    Server maintains K models.

    Each round:
      1. sample clients
      2. each client chooses the model with minimum local loss
      3. client trains from that model
      4. server aggregates updates within each chosen cluster
    """
    rng = np.random.default_rng(seed)

    models = []

    for c in range(k):
        set_seed(seed + 7000 + c)
        model = SmallCNN().cpu()
        models.append(model)

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
        round_assignments = []

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

            round_assignments.append(chosen_c)

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
                avg_state = weighted_average_states(
                    states=cluster_states[c],
                    weights=cluster_weights[c],
                )
                models[c].load_state_dict(avg_state)

        unique, counts = np.unique(round_assignments, return_counts=True)
        assign_info = ", ".join(
            [f"c{int(c)}:{int(n)}" for c, n in zip(unique, counts)]
        )

        print(f"[IFCA K={k}] Round {r + 1}/{rounds} done | {assign_info}")

    return [m.cpu() for m in models]


# -----------------------------
# 5. Evaluation
# -----------------------------

def evaluate_ifca_models(
    models: List[nn.Module],
    clients: List[ClientData],
    train_dataset,
    test_dataset,
    batch_size: int,
    device,
    num_workers: int,
) -> Dict[str, float]:
    """
    Evaluation:
    1. assign each client to best IFCA model using training loss
    2. evaluate selected model on that client's test data
    """
    assignments = assign_all_clients(
        models=models,
        clients=clients,
        train_dataset=train_dataset,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
    )

    unique, counts = np.unique(assignments, return_counts=True)

    print("[IFCA Eval] Final client assignments:")
    for c, n in zip(unique, counts):
        print(f"  cluster {int(c)}: {int(n)} clients")

    client_accs = []
    client_macro_f1s = []

    all_true = []
    all_pred = []

    total_correct = 0
    total_count = 0

    for i, client in enumerate(clients):
        c = int(assignments[i])
        model = models[c]

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
        macro_f1 = f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )

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
        "k_pred": int(len(np.unique(assignments))),
        "client_avg_acc": float(np.mean(client_accs)),
        "micro_acc": float(total_correct / max(total_count, 1)),
        "client_avg_macro_f1": float(np.mean(client_macro_f1s)),
        "global_macro_f1": float(global_macro_f1),
    }


# -----------------------------
# 6. Main
# -----------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--output", type=str, default="ifca_results.csv")

    parser.add_argument("--num-clients", type=int, default=100)
    parser.add_argument("--train-samples-per-client", type=int, default=400)
    parser.add_argument("--test-samples-per-client", type=int, default=100)
    parser.add_argument("--major-ratio", type=float, default=0.85)

    parser.add_argument("--k-list", type=str, default="2,3,4,5,6,8")

    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--client-frac", type=float, default=0.2)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.02)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )

    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)
    device = get_device(args.device)

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

    k_list = [int(x.strip()) for x in args.k_list.split(",") if x.strip()]

    results = []

    for k in k_list:
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
            seed=args.seed + 1000 + k,
            num_workers=args.num_workers,
        )

        metrics = evaluate_ifca_models(
            models=ifca_models,
            clients=clients,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            batch_size=args.batch_size,
            device=device,
            num_workers=args.num_workers,
        )

        method_name = "IFCA-oracleK" if k == 4 else "IFCA"

        row = {
            "method": method_name,
            "k_setting": k,
            **metrics,
        }

        results.append(row)

        print(f"[IFCA K={k}] {metrics}")

    df = pd.DataFrame(results)

    print("\n========== Final IFCA results ==========")
    print(df)

    df.to_csv(args.output, index=False)
    print(f"\nSaved results to: {args.output}")


if __name__ == "__main__":
    main()
