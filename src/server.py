"""
Federated Server (CARES-Lite)
==============================
Three-phase pipeline:

  Phase 0  — 初始化全局模型 + 超参数
  Phase 1  — Warm-up 全局 FedAvg（前 T_warm 轮）
  过渡      — 全量 profiling + 首次 DPMM 聚类 + 初始化组模型
  Phase 2  — 分组 FedAvg（T_warm 轮后至收敛）+ 每 τ 轮动态重聚类

修改点：
  1. 训练阶段 client 仍使用所属组模型。
  2. 动态重聚类阶段默认使用统一的 EMA global anchor model 做 profiling。
  3. 保留 probe_anchor="assigned" 作为旧逻辑 ablation。
  4. 训练轮中默认不再计算 loss profile，只在 full profiling / re-clustering 时计算。
  5. 同一次 profiling 内所有 clients 共享相同 random probe seed，保证 profile 维度可比较。
  6. 新增论文常用检测指标：ACC、Precision、Recall、F1。
  7. Server 端模型与 anchor state 始终保留在 CPU，避免 CPU/CUDA 混用。
"""

import copy
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
)
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler

from src.client import FLClient
from src.data import set_seed
from src.model import SmallCNN


class FLServer:
    """Central server coordinating the CARES-Lite FL pipeline."""

    def __init__(
        self,
        clients: List[FLClient],
        device: torch.device,
        seed: int,
        model_fn: Callable[[], nn.Module] = SmallCNN,
    ):
        self.clients = clients
        self.device = device
        self.seed = seed
        self.model_fn = model_fn
        self.rng = np.random.default_rng(seed)
        self.num_clients = len(clients)

        self.global_model: Optional[nn.Module] = None
        self.cluster_models: Dict[int, nn.Module] = {}

        self.assignments = np.zeros(
            self.num_clients,
            dtype=int,
        )

        self.k_pred: int = 0
        self.client_profiles: Dict[int, np.ndarray] = {}

        self.anchor_state: Optional[Dict[str, torch.Tensor]] = None

    @staticmethod
    def _state_to_cpu(
        state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Return a detached CPU copy of a state dictionary."""
        return {
            key: value.detach().cpu().clone()
            for key, value in state.items()
        }

    def _select_clients(self, frac: float) -> List[int]:
        m = max(1, int(round(frac * self.num_clients)))

        return self.rng.choice(
            self.num_clients,
            size=m,
            replace=False,
        ).tolist()

    @staticmethod
    def _aggregate_deltas(
        deltas: List[Dict[str, torch.Tensor]],
        weights: List[int],
    ) -> Dict[str, torch.Tensor]:
        """Weighted average of client deltas, always returned on CPU."""
        if not deltas:
            raise ValueError("Cannot aggregate an empty delta list.")

        if len(deltas) != len(weights):
            raise ValueError("deltas and weights must have equal length.")

        total = float(sum(weights))

        if total <= 0:
            raise ValueError("Sum of aggregation weights must be positive.")

        cpu_deltas = [
            FLServer._state_to_cpu(delta)
            for delta in deltas
        ]

        avg: Dict[str, torch.Tensor] = {}

        for key in cpu_deltas[0]:
            if torch.is_floating_point(cpu_deltas[0][key]):
                avg[key] = sum(
                    delta[key].float() * (weight / total)
                    for delta, weight in zip(cpu_deltas, weights)
                ).to(dtype=cpu_deltas[0][key].dtype)
            else:
                avg[key] = torch.zeros_like(cpu_deltas[0][key])

        return avg

    @staticmethod
    def _apply_delta(
        state: Dict[str, torch.Tensor],
        delta: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Apply a CPU delta to a CPU model state."""
        state_cpu = FLServer._state_to_cpu(state)
        delta_cpu = FLServer._state_to_cpu(delta)

        new_state = copy.deepcopy(state_cpu)

        for key in new_state:
            if torch.is_floating_point(new_state[key]):
                new_state[key] = (
                    state_cpu[key].float()
                    + delta_cpu[key].float()
                ).to(dtype=state_cpu[key].dtype)
            else:
                new_state[key] = state_cpu[key].clone()

        return new_state

    def _full_profiling(
        self,
        model_state_fn: Callable[[int], Dict[str, torch.Tensor]],
        M: int,
        sigma: float,
        probe_seed: int,
    ):
        print(
            f"  [Server] Full profiling for {self.num_clients} clients "
            f"(probe_seed={probe_seed}) ..."
        )

        for i, client in enumerate(self.clients):
            state = self._state_to_cpu(model_state_fn(i))

            profile = client.compute_loss_profile(
                state,
                self.model_fn,
                M,
                sigma,
                probe_seed,
            )

            self.client_profiles[i] = profile

            if (i + 1) % 20 == 0:
                print(f"    profiled {i + 1}/{self.num_clients}")

        self._print_profile_diagnostics()

    def _print_profile_diagnostics(self):
        """打印 loss profile 统计信息，用于验证 probe 是否产生了有效信号。"""
        Z = self._get_profile_matrix()

        print(f"\n  [Diagnostics] Profile matrix shape: {Z.shape}")
        print("  [Diagnostics] Per-dimension stats (across clients):")
        print(f"    mean:  {Z.mean(axis=0)[:5].round(3)} ...")
        print(f"    std:   {Z.std(axis=0)[:5].round(3)} ...")
        print(
            f"    range: "
            f"{(Z.max(axis=0) - Z.min(axis=0))[:5].round(3)} ..."
        )

        inter_client_std = Z.std(axis=0).mean()
        intra_client_std = Z.std(axis=1).mean()

        print(
            f"  [Diagnostics] Avg inter-client std (across dims): "
            f"{inter_client_std:.4f}"
        )
        print(
            f"  [Diagnostics] Avg intra-client std (across probes): "
            f"{intra_client_std:.4f}"
        )

        if inter_client_std < 0.01:
            print(
                "  WARNING: Inter-client std very low -> "
                "probes may lack diversity!"
            )
        else:
            print("  Profile diversity looks reasonable")

    def _get_profile_matrix(self) -> np.ndarray:
        M = len(next(iter(self.client_profiles.values())))

        Z = np.zeros(
            (self.num_clients, M),
            dtype=np.float32,
        )

        for i, profile in self.client_profiles.items():
            Z[i] = profile

        return Z

    def _normalize_profiles(
        self,
        Z: np.ndarray,
        clip_norm: float,
        noise_sigma: float,
    ) -> np.ndarray:
        rng = np.random.default_rng(self.seed + 4000)

        R = Z.copy().astype(np.float32)

        mu = R.mean(axis=1, keepdims=True)
        std = R.std(axis=1, keepdims=True) + 1e-6

        R = (R - mu) / std

        norms = np.linalg.norm(
            R,
            axis=1,
            keepdims=True,
        ) + 1e-12

        R *= np.minimum(1.0, clip_norm / norms)

        if noise_sigma > 0:
            R += rng.normal(
                0.0,
                noise_sigma * clip_norm,
                size=R.shape,
            ).astype(np.float32)

        return R

    def _run_dpmm(
        self,
        profiles: np.ndarray,
        max_components: int,
        prior: float,
    ):
        X = StandardScaler().fit_transform(profiles)

        dpmm = BayesianGaussianMixture(
            n_components=max_components,
            covariance_type="diag",
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=prior,
            max_iter=1000,
            n_init=10,
            random_state=self.seed + 5000,
            init_params="kmeans",
        )

        self.assignments = dpmm.fit_predict(X)
        self.k_pred = int(len(np.unique(self.assignments)))

    def _merge_small_clusters(
        self,
        min_size: int,
        profiles: np.ndarray,
    ):
        cluster_map = self._get_cluster_map()

        centers: Dict[int, np.ndarray] = {}

        for gid, members in cluster_map.items():
            centers[gid] = profiles[members].mean(axis=0)

        changed = True

        while changed:
            changed = False
            cluster_map = self._get_cluster_map()

            small_groups = [
                gid
                for gid, members in cluster_map.items()
                if len(members) < min_size
            ]

            big_groups = [
                gid
                for gid, members in cluster_map.items()
                if len(members) >= min_size
            ]

            if not big_groups and small_groups:
                biggest = max(
                    small_groups,
                    key=lambda gid: len(cluster_map[gid]),
                )

                big_groups = [biggest]
                small_groups = [
                    gid
                    for gid in small_groups
                    if gid != biggest
                ]

            for small_gid in small_groups:
                if not big_groups:
                    break

                small_center = centers.get(
                    small_gid,
                    profiles[cluster_map[small_gid]].mean(axis=0),
                )

                distances = {
                    big_gid: np.linalg.norm(
                        small_center
                        - centers.get(
                            big_gid,
                            profiles[cluster_map[big_gid]].mean(axis=0),
                        )
                    )
                    for big_gid in big_groups
                }

                nearest_gid = min(distances, key=distances.get)

                for client_idx in cluster_map[small_gid]:
                    self.assignments[client_idx] = nearest_gid

                changed = True

                new_members = (
                    cluster_map[nearest_gid]
                    + cluster_map[small_gid]
                )

                centers[nearest_gid] = profiles[new_members].mean(axis=0)

        self._relabel_assignments()
        self.k_pred = int(len(np.unique(self.assignments)))

    def _relabel_assignments(self):
        """Relabel cluster IDs to contiguous integers: 0, 1, 2, ..."""
        old_labels = sorted(np.unique(self.assignments).tolist())

        mapping = {
            old_label: new_label
            for new_label, old_label in enumerate(old_labels)
        }

        self.assignments = np.array(
            [mapping[int(label)] for label in self.assignments],
            dtype=int,
        )

    def _get_cluster_map(self) -> Dict[int, List[int]]:
        """Return mapping: cluster_id -> list of client indices."""
        mapping: Dict[int, List[int]] = {}

        for client_idx, cluster_id in enumerate(self.assignments):
            mapping.setdefault(int(cluster_id), []).append(client_idx)

        return mapping

    def _weighted_avg_cluster_models(self) -> Dict[str, torch.Tensor]:
        """
        Build a global aggregated anchor model by weighted-averaging
        current group models.

        Weight = number of clients currently assigned to each group.
        This anchor is used only for profiling / re-clustering.
        """
        if not self.cluster_models:
            if self.global_model is None:
                raise RuntimeError(
                    "No global model or cluster models available."
                )

            return self._state_to_cpu(self.global_model.state_dict())

        cluster_map = self._get_cluster_map()

        states: List[Dict[str, torch.Tensor]] = []
        weights: List[int] = []

        for gid, model in self.cluster_models.items():
            n_clients = len(cluster_map.get(int(gid), []))

            if n_clients <= 0:
                continue

            model.cpu()

            states.append(
                self._state_to_cpu(model.state_dict())
            )

            weights.append(n_clients)

        if not states:
            if self.global_model is None:
                raise RuntimeError(
                    "No valid cluster states for anchor update."
                )

            return self._state_to_cpu(self.global_model.state_dict())

        return self._weighted_avg_states(states, weights)


    def _update_anchor_state(
        self,
        beta: float = 0.9,
    ) -> Dict[str, torch.Tensor]:
        """
        EMA update of the global profiling anchor.

            anchor <- beta * anchor
                      + (1 - beta) * weighted_avg(cluster_models)

        All anchor tensors are explicitly kept on CPU.
        """
        new_anchor = self._state_to_cpu(
            self._weighted_avg_cluster_models()
        )

        if self.anchor_state is None:
            self.anchor_state = copy.deepcopy(new_anchor)
        else:
            old_anchor = self._state_to_cpu(self.anchor_state)
            updated_anchor: Dict[str, torch.Tensor] = {}

            for key in new_anchor:
                if torch.is_floating_point(new_anchor[key]):
                    updated_anchor[key] = (
                        beta * old_anchor[key].float()
                        + (1.0 - beta) * new_anchor[key].float()
                    ).to(dtype=new_anchor[key].dtype)
                else:
                    updated_anchor[key] = new_anchor[key].clone()

            self.anchor_state = updated_anchor

        return self._state_to_cpu(self.anchor_state)

    def _get_probe_anchor_state(
        self,
        probe_anchor: str,
        anchor_ema_beta: float,
    ) -> Dict[str, torch.Tensor]:
        """Return the base model used for re-clustering profiling."""
        if probe_anchor == "global":
            return self._state_to_cpu(
                self._weighted_avg_cluster_models()
            )

        if probe_anchor == "ema_global":
            if self.anchor_state is None:
                return self._update_anchor_state(
                    beta=anchor_ema_beta
                )

            return self._state_to_cpu(self.anchor_state)

        raise ValueError(f"Unsupported probe_anchor: {probe_anchor}")
    @staticmethod
    def _all_class_labels(num_classes: int) -> List[int]:
        """Complete class set for fixed-label metrics."""
        return list(range(int(num_classes)))

    @staticmethod
    def _fixed_macro_f1(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        num_classes: int,
    ) -> float:
        """Macro-F1 over all classes, including locally absent classes."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.size == 0:
            return 0.0
        return float(
            f1_score(
                y_true,
                y_pred,
                labels=FLServer._all_class_labels(num_classes),
                average="macro",
                zero_division=0,
            )
        )

    @staticmethod
    def _paper_classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        num_classes: int,
    ) -> Dict[str, float]:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if y_true.size == 0:
            return {
                "acc": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "tn": 0,
                "fp": 0,
                "fn": 0,
                "tp": 0,
            }

        acc = float(np.mean(y_true == y_pred))

        if int(num_classes) == 2:
            # Binary NIDS paper metrics:
            # report attack/positive-class precision, recall, and F1.
            precision = float(
                precision_score(
                    y_true,
                    y_pred,
                    pos_label=1,
                    average="binary",
                    zero_division=0,
                )
            )
            recall = float(
                recall_score(
                    y_true,
                    y_pred,
                    pos_label=1,
                    average="binary",
                    zero_division=0,
                )
            )
            f1 = float(
                f1_score(
                    y_true,
                    y_pred,
                    pos_label=1,
                    average="binary",
                    zero_division=0,
                )
            )
            tn = int(np.sum((y_true == 0) & (y_pred == 0)))
            fp = int(np.sum((y_true == 0) & (y_pred == 1)))
            fn = int(np.sum((y_true == 1) & (y_pred == 0)))
            tp = int(np.sum((y_true == 1) & (y_pred == 1)))

        else:
            # Multiclass metrics:
            # use fixed-label macro averaging over the full class set.
            labels = FLServer._all_class_labels(num_classes)
            precision = float(
                precision_score(
                    y_true,
                    y_pred,
                    labels=labels,
                    average="macro",
                    zero_division=0,
                )
            )
            recall = float(
                recall_score(
                    y_true,
                    y_pred,
                    labels=labels,
                    average="macro",
                    zero_division=0,
                )
            )
            f1 = float(
                f1_score(
                    y_true,
                    y_pred,
                    labels=labels,
                    average="macro",
                    zero_division=0,
                )
            )
            tn, fp, fn, tp = 0, 0, 0, 0

        return {
            "acc": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }


    def _cluster_quality_metrics(self) -> Dict[str, float]:
        """
        Compute clustering quality against ground-truth groups.

        These metrics are only for synthetic/controlled experiments where
        client.group_id is available. They are not used by CARES itself.
        """
        y_true = np.array(
            [
                int(self.clients[i].group_id)
                for i in range(self.num_clients)
            ],
            dtype=int,
        )

        y_pred = self.assignments.astype(int)

        ari = adjusted_rand_score(y_true, y_pred)

        nmi = normalized_mutual_info_score(
            y_true,
            y_pred,
        )

        purity_correct = 0

        for _, members in self._get_cluster_map().items():
            labels = y_true[members]

            if len(labels) == 0:
                continue

            _, counts = np.unique(
                labels,
                return_counts=True,
            )

            purity_correct += int(counts.max())

        purity = purity_correct / max(self.num_clients, 1)

        return {
            "ari": float(ari),
            "nmi": float(nmi),
            "purity": float(purity),
        }

    def _init_group_models(self):
        """Initialize each cluster model from the warm-up global model."""
        if self.global_model is None:
            raise RuntimeError("Global model is not initialized.")

        self.global_model.cpu()

        global_state = self._state_to_cpu(
            self.global_model.state_dict()
        )

        cluster_map = self._get_cluster_map()
        self.cluster_models = {}

        for gid in sorted(cluster_map):
            model = self.model_fn()
            model.load_state_dict(copy.deepcopy(global_state))
            model.cpu()

            self.cluster_models[gid] = model

        print(
            f"  [Server] Initialized {len(self.cluster_models)} group models "
            f"from global model"
        )

    def _reinit_group_models(
        self,
        old_assignments: np.ndarray,
    ):
        """Rebuild group models after dynamic re-clustering."""
        old_model_states = {
            gid: self._state_to_cpu(model.state_dict())
            for gid, model in self.cluster_models.items()
        }

        new_cluster_map = self._get_cluster_map()
        new_models: Dict[int, nn.Module] = {}

        for new_gid, member_ids in new_cluster_map.items():
            old_group_counts: Dict[int, int] = {}

            for member_id in member_ids:
                old_gid = int(old_assignments[member_id])

                old_group_counts[old_gid] = (
                    old_group_counts.get(old_gid, 0) + 1
                )

            states: List[Dict[str, torch.Tensor]] = []
            weights: List[int] = []

            for old_gid, count in old_group_counts.items():
                if old_gid in old_model_states:
                    states.append(old_model_states[old_gid])
                    weights.append(count)

            if states:
                avg_state = self._weighted_avg_states(
                    states,
                    weights,
                )
            elif old_model_states:
                avg_state = self._state_to_cpu(
                    next(iter(old_model_states.values()))
                )
            elif self.global_model is not None:
                avg_state = self._state_to_cpu(
                    self.global_model.state_dict()
                )
            else:
                raise RuntimeError(
                    "Cannot initialize group models after re-clustering."
                )

            model = self.model_fn()
            model.load_state_dict(avg_state)
            model.cpu()

            new_models[new_gid] = model

        self.cluster_models = new_models

        print(
            f"  [Server] Matched and inherited {len(self.cluster_models)} "
            f"group models after re-clustering"
        )

    @staticmethod
    def _weighted_avg_states(
        states: List[Dict[str, torch.Tensor]],
        weights: List[int],
    ) -> Dict[str, torch.Tensor]:
        """Weighted average of model state dicts, returned on CPU."""
        if not states:
            raise ValueError("Cannot average an empty state list.")

        if len(states) != len(weights):
            raise ValueError("states and weights must have equal length.")

        total = float(sum(weights))

        if total <= 0:
            raise ValueError("Sum of state weights must be positive.")

        cpu_states = [
            FLServer._state_to_cpu(state)
            for state in states
        ]

        avg = copy.deepcopy(cpu_states[0])

        for key in avg:
            if torch.is_floating_point(avg[key]):
                avg[key] = sum(
                    state[key].float() * (weight / total)
                    for state, weight in zip(cpu_states, weights)
                ).to(dtype=cpu_states[0][key].dtype)
            else:
                avg[key] = cpu_states[0][key].clone()

        return avg

    def evaluate_global_model(self) -> Dict[str, float]:
        """Evaluate the warm-up global model before clustering."""
        if self.global_model is None:
            raise RuntimeError("Global model is not initialized.")

        accs = []
        macro_f1s = []
        client_precisions = []
        client_recalls = []
        client_f1s = []

        all_y_true = []
        all_y_pred = []

        total_correct = 0
        total_n = 0

        num_classes = self.clients[0].num_classes if self.clients else 2

        for client in self.clients:
            y_true, y_pred = client.evaluate(self.global_model)

            n_samples = len(y_true)
            correct = int((y_true == y_pred).sum())

            client_metrics = self._paper_classification_metrics(
                y_true,
                y_pred,
                num_classes=num_classes,
            )

            accs.append(client_metrics["acc"])
            client_precisions.append(client_metrics["precision"])
            client_recalls.append(client_metrics["recall"])
            client_f1s.append(client_metrics["f1"])

            macro_f1s.append(
                self._fixed_macro_f1(y_true, y_pred, num_classes)
                if n_samples > 0
                else 0.0
            )


            all_y_true.extend(y_true.tolist())
            all_y_pred.extend(y_pred.tolist())

            total_correct += correct
            total_n += n_samples

        all_y_true_arr = np.asarray(all_y_true)
        all_y_pred_arr = np.asarray(all_y_pred)

        paper_metrics = self._paper_classification_metrics(
            all_y_true_arr,
            all_y_pred_arr,
            num_classes=num_classes,
        )

        global_macro_f1 = (
            self._fixed_macro_f1(
                all_y_true_arr,
                all_y_pred_arr,
                num_classes,
            )
            if total_n > 0
            else 0.0
        )


        return {
            # Paper metrics: pooled globally across every client test sample.
            "acc": paper_metrics["acc"],
            "precision": paper_metrics["precision"],
            "recall": paper_metrics["recall"],
            "f1": paper_metrics["f1"],

            # Binary confusion matrix.
            "tn": paper_metrics["tn"],
            "fp": paper_metrics["fp"],
            "fn": paper_metrics["fn"],
            "tp": paper_metrics["tp"],

            # Original metrics kept for compatibility.
            "client_avg_acc": float(np.mean(accs)) if accs else 0.0,
            "micro_acc": float(total_correct / max(total_n, 1)),
            "client_avg_macro_f1": (
                float(np.mean(macro_f1s))
                if macro_f1s
                else 0.0
            ),
            "global_macro_f1": global_macro_f1,

            # Optional client-average paper metrics.
            "client_avg_precision": (
                float(np.mean(client_precisions))
                if client_precisions
                else 0.0
            ),
            "client_avg_recall": (
                float(np.mean(client_recalls))
                if client_recalls
                else 0.0
            ),
            "client_avg_f1": (
                float(np.mean(client_f1s))
                if client_f1s
                else 0.0
            ),
        }

    def evaluate(self) -> Dict[str, float]:
        """Evaluate clustered models on all client test sets."""
        accs = []
        macro_f1s = []
        client_precisions = []
        client_recalls = []
        client_f1s = []

        all_y_true = []
        all_y_pred = []

        total_correct = 0
        total_n = 0

        num_classes = self.clients[0].num_classes if self.clients else 2

        for i, client in enumerate(self.clients):
            gid = int(self.assignments[i])
            model = self.cluster_models[gid]

            y_true, y_pred = client.evaluate(model)

            n_samples = len(y_true)
            correct = int((y_true == y_pred).sum())

            client_metrics = self._paper_classification_metrics(
                y_true,
                y_pred,
                num_classes=num_classes,
            )

            accs.append(client_metrics["acc"])
            client_precisions.append(client_metrics["precision"])
            client_recalls.append(client_metrics["recall"])
            client_f1s.append(client_metrics["f1"])

            macro_f1s.append(
                self._fixed_macro_f1(y_true, y_pred, num_classes)
                if n_samples > 0
                else 0.0
            )


            all_y_true.extend(y_true.tolist())
            all_y_pred.extend(y_pred.tolist())

            total_correct += correct
            total_n += n_samples

        all_y_true_arr = np.asarray(all_y_true)
        all_y_pred_arr = np.asarray(all_y_pred)

        paper_metrics = self._paper_classification_metrics(
            all_y_true_arr,
            all_y_pred_arr,
            num_classes=num_classes,
        )

        global_macro_f1 = (
            self._fixed_macro_f1(
                all_y_true_arr,
                all_y_pred_arr,
                num_classes,
            )
            if total_n > 0
            else 0.0
        )

        metrics = {
            "k_pred": self.k_pred,

            # Paper metrics: pooled globally across every client test sample.
            "acc": paper_metrics["acc"],
            "precision": paper_metrics["precision"],
            "recall": paper_metrics["recall"],
            "f1": paper_metrics["f1"],

            # Binary confusion matrix.
            "tn": paper_metrics["tn"],
            "fp": paper_metrics["fp"],
            "fn": paper_metrics["fn"],
            "tp": paper_metrics["tp"],

            # Original metrics kept for compatibility.
            "client_avg_acc": float(np.mean(accs)) if accs else 0.0,
            "micro_acc": float(total_correct / max(total_n, 1)),
            "client_avg_macro_f1": (
                float(np.mean(macro_f1s))
                if macro_f1s
                else 0.0
            ),
            "global_macro_f1": global_macro_f1,

            # Optional client-average paper metrics.
            "client_avg_precision": (
                float(np.mean(client_precisions))
                if client_precisions
                else 0.0
            ),
            "client_avg_recall": (
                float(np.mean(client_recalls))
                if client_recalls
                else 0.0
            ),
            "client_avg_f1": (
                float(np.mean(client_f1s))
                if client_f1s
                else 0.0
            ),
        }

        metrics.update(self._cluster_quality_metrics())

        return metrics

    def _print_eval_snapshot(self, tag: str):
        """Print an evaluation snapshot during clustered training."""
        metrics = self.evaluate()

        print(
            f"  [Eval:{tag}] "
            f"K={metrics['k_pred']} | "
            f"ACC={metrics['acc']:.4f} | "
            f"Precision={metrics['precision']:.4f} | "
            f"Recall={metrics['recall']:.4f} | "
            f"F1={metrics['f1']:.4f} | "
            f"GlobalMacroF1={metrics['global_macro_f1']:.4f} | "
            f"ClientAvgMacroF1={metrics['client_avg_macro_f1']:.4f} | "
            f"ARI={metrics.get('ari', float('nan')):.4f} | "
            f"NMI={metrics.get('nmi', float('nan')):.4f} | "
            f"Purity={metrics.get('purity', float('nan')):.4f}"
        )

    def _print_cluster_info(self):
        print(f"  [Server] K = {self.k_pred}")

        for cluster_id, members in sorted(
            self._get_cluster_map().items()
        ):
            group_counts: Dict[int, int] = {}

            for client_idx in members:
                group_id = self.clients[client_idx].group_id

                group_counts[group_id] = (
                    group_counts.get(group_id, 0) + 1
                )

            group_str = ", ".join(
                f"G{group_id}:{count}"
                for group_id, count in sorted(group_counts.items())
            )

            print(
                f"    cluster {cluster_id}: "
                f"{len(members)} clients [{group_str}]"
            )

    def run(
        self,
        *,
        total_rounds: int,
        warmup_rounds: int,
        cluster_interval: int,
        min_cluster_size: int,
        probe_pool_size: int,
        probe_sigma: float,
        clip_norm: float,
        noise_sigma: float,
        dpmm_max_comp: int,
        dpmm_prior: float,
        client_frac: float,
        local_epochs: int,
        lr: float,
        probe_anchor: str = "ema_global",
        anchor_ema_beta: float = 0.9,
        profile_during_training: bool = False,
    ) -> Dict[str, float]:

        print("\n" + "=" * 55)
        print(" Phase 0: Initialization")
        print("=" * 55)

        if probe_anchor not in {
            "assigned",
            "global",
            "ema_global",
        }:
            raise ValueError(
                "probe_anchor must be one of: "
                "assigned, global, ema_global"
            )

        if cluster_interval <= 0:
            raise ValueError("cluster_interval must be positive.")

        set_seed(self.seed)

        # Server-side global model is intentionally stored on CPU.
        self.global_model = self.model_fn()
        self.global_model.cpu()

        print("  Global model initialized")
        print(f"  Total rounds: {total_rounds}")
        print(f"  Warm-up rounds: {warmup_rounds}")
        print(f"  Cluster interval tau: {cluster_interval}")

        n_ablation = min(
            probe_pool_size - 1,
            self.clients[0].num_classes,
        )

        print(
            f"  Probe pool M: {probe_pool_size}, "
            f"sigma: {probe_sigma}"
        )

        print(
            f"  Probes: 1 original + {n_ablation} class-ablation + "
            f"{max(0, probe_pool_size - 1 - n_ablation)} random"
        )

        print(f"  Probe anchor: {probe_anchor}")

        if probe_anchor == "ema_global":
            print(f"  Anchor EMA beta: {anchor_ema_beta}")

        print(
            f"  Profile during training rounds: "
            f"{profile_during_training}"
        )

        print("\n" + "=" * 55)
        print(" Phase 1: Warm-up Global FedAvg")
        print("=" * 55)

        for t in range(warmup_rounds):
            selected = self._select_clients(client_frac)

            global_state = self._state_to_cpu(
                self.global_model.state_dict()
            )

            deltas: List[Dict[str, torch.Tensor]] = []
            weights: List[int] = []

            for idx in selected:
                client = self.clients[idx]

                if profile_during_training:
                    profile = client.compute_loss_profile(
                        global_state,
                        self.model_fn,
                        probe_pool_size,
                        probe_sigma,
                        self.seed + 8000 * t,
                    )

                    self.client_profiles[idx] = profile

                delta = client.local_train(
                    global_state,
                    self.model_fn,
                    local_epochs,
                    lr,
                )

                deltas.append(delta)
                weights.append(client.num_train)

            avg_delta = self._aggregate_deltas(
                deltas,
                weights,
            )

            new_state = self._apply_delta(
                global_state,
                avg_delta,
            )

            self.global_model.cpu()
            self.global_model.load_state_dict(new_state)

            print(
                f"  [Warmup] Round {t + 1}/{warmup_rounds} done "
                f"(selected {len(selected)} clients)"
            )

            warm_metrics = self.evaluate_global_model()

            print(
                f"  [Eval:Warmup R{t + 1}] "
                f"ACC={warm_metrics['acc']:.4f} | "
                f"Precision={warm_metrics['precision']:.4f} | "
                f"Recall={warm_metrics['recall']:.4f} | "
                f"F1={warm_metrics['f1']:.4f} | "
                f"GlobalMacroF1={warm_metrics['global_macro_f1']:.4f}"
            )

        print("\n" + "=" * 55)
        print(" Transition: First DPMM Clustering")
        print("=" * 55)

        self._full_profiling(
            model_state_fn=lambda _: self._state_to_cpu(
                self.global_model.state_dict()
            ),
            M=probe_pool_size,
            sigma=probe_sigma,
            probe_seed=self.seed + 9100,
        )

        Z = self._get_profile_matrix()

        profiles_normed = self._normalize_profiles(
            Z,
            clip_norm,
            noise_sigma,
        )

        self._run_dpmm(
            profiles_normed,
            dpmm_max_comp,
            dpmm_prior,
        )

        self._merge_small_clusters(
            min_cluster_size,
            profiles_normed,
        )

        self._print_cluster_info()
        self._init_group_models()

        self.anchor_state = self._state_to_cpu(
            self.global_model.state_dict()
        )

        print(
            "  [Server] Anchor model initialized from warm-up global model"
        )

        print("\n" + "=" * 55)
        print(" Phase 2: Clustered FedAvg with Dynamic Re-clustering")
        print("=" * 55)

        clustered_rounds = total_rounds - warmup_rounds

        for t_c in range(clustered_rounds):
            t_global = warmup_rounds + t_c

            if t_c > 0 and t_c % cluster_interval == 0:
                print(
                    f"\n  [Server] Re-clustering at round "
                    f"{t_global + 1} ..."
                )

                old_assignments = self.assignments.copy()

                if probe_anchor == "assigned":
                    print(
                        "  [Server] Profiling base: assigned group models"
                    )

                    self._full_profiling(
                        model_state_fn=lambda i: self._state_to_cpu(
                            self.cluster_models[
                                int(self.assignments[i])
                            ].state_dict()
                        ),
                        M=probe_pool_size,
                        sigma=probe_sigma,
                        probe_seed=self.seed + 9100 + t_global,
                    )

                else:
                    print(
                        f"  [Server] Profiling base: "
                        f"{probe_anchor} anchor model"
                    )

                    anchor_state = self._get_probe_anchor_state(
                        probe_anchor=probe_anchor,
                        anchor_ema_beta=anchor_ema_beta,
                    )

                    self._full_profiling(
                        model_state_fn=lambda _: self._state_to_cpu(
                            anchor_state
                        ),
                        M=probe_pool_size,
                        sigma=probe_sigma,
                        probe_seed=self.seed + 9100 + t_global,
                    )

                Z = self._get_profile_matrix()

                profiles_normed = self._normalize_profiles(
                    Z,
                    clip_norm,
                    noise_sigma,
                )

                self._run_dpmm(
                    profiles_normed,
                    dpmm_max_comp,
                    dpmm_prior,
                )

                self._merge_small_clusters(
                    min_cluster_size,
                    profiles_normed,
                )

                self._print_cluster_info()

                self._reinit_group_models(old_assignments)

                if probe_anchor == "ema_global":
                    self._update_anchor_state(
                        beta=anchor_ema_beta
                    )

            selected = self._select_clients(client_frac)

            group_deltas: Dict[int, Tuple[List, List]] = {
                gid: ([], [])
                for gid in self.cluster_models
            }

            for idx in selected:
                client = self.clients[idx]

                gid = int(self.assignments[idx])

                self.cluster_models[gid].cpu()

                group_state = self._state_to_cpu(
                    self.cluster_models[gid].state_dict()
                )

                if profile_during_training:
                    profile = client.compute_loss_profile(
                        group_state,
                        self.model_fn,
                        probe_pool_size,
                        probe_sigma,
                        self.seed + 8000 * t_global,
                    )

                    self.client_profiles[idx] = profile

                delta = client.local_train(
                    group_state,
                    self.model_fn,
                    local_epochs,
                    lr,
                )

                group_deltas[gid][0].append(delta)
                group_deltas[gid][1].append(client.num_train)

            for gid in self.cluster_models:
                group_delta_list, group_weight_list = group_deltas[gid]

                if not group_delta_list:
                    continue

                avg_delta = self._aggregate_deltas(
                    group_delta_list,
                    group_weight_list,
                )

                self.cluster_models[gid].cpu()

                old_group_state = self._state_to_cpu(
                    self.cluster_models[gid].state_dict()
                )

                new_group_state = self._apply_delta(
                    old_group_state,
                    avg_delta,
                )

                self.cluster_models[gid].load_state_dict(
                    new_group_state
                )

            if probe_anchor == "ema_global":
                self._update_anchor_state(beta=anchor_ema_beta)

            if (t_c + 1) % 5 == 0 or t_c == 0:
                print(
                    f"  [Clustered] Round {t_global + 1}/{total_rounds} "
                    f"done (selected {len(selected)} clients)"
                )

                self._print_eval_snapshot(
                    f"R{t_global + 1}"
                )

        print("\n" + "=" * 55)
        print(" Evaluation")
        print("=" * 55)

        return self.evaluate()
