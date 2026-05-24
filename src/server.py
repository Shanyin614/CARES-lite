"""
Federated Server (CARES-Lite)
==============================
Three-phase pipeline:

  Phase 0  — 初始化全局模型 + 超参数
  Phase 1  — Warm-up 全局 FedAvg（前 T_warm 轮）
  过渡      — 全量 profiling + 首次 DPMM 聚类 + 初始化组模型
  Phase 2  — 分组 FedAvg（T_warm 轮后至收敛）+ 每 τ 轮动态重聚类
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
)

from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler

from src.data import set_seed
from src.model import SmallCNN
from src.client import FLClient


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
        self.assignments: np.ndarray = np.zeros(self.num_clients, dtype=int)
        self.k_pred: int = 0
        self.client_profiles: Dict[int, np.ndarray] = {}

    # ══════════════════════════════════════════════════════
    #  Client selection
    # ══════════════════════════════════════════════════════

    def _select_clients(self, frac: float) -> List[int]:
        m = max(1, int(round(frac * self.num_clients)))
        return self.rng.choice(
            self.num_clients, size=m, replace=False,
        ).tolist()

    # ══════════════════════════════════════════════════════
    #  Delta aggregation
    # ══════════════════════════════════════════════════════

    @staticmethod
    def _aggregate_deltas(
        deltas: List[Dict[str, torch.Tensor]],
        weights: List[int],
    ) -> Dict[str, torch.Tensor]:
        total = float(sum(weights))
        avg: Dict[str, torch.Tensor] = {}
        for key in deltas[0]:
            avg[key] = sum(
                d[key].float() * (w / total) for d, w in zip(deltas, weights)
            )
        return avg

    @staticmethod
    def _apply_delta(
        state: Dict[str, torch.Tensor],
        delta: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        new_state = copy.deepcopy(state)
        for key in new_state:
            new_state[key] = new_state[key].float() + delta[key].float()
        return new_state

    # ══════════════════════════════════════════════════════
    #  Full profiling
    # ══════════════════════════════════════════════════════

    def _full_profiling(
        self,
        model_state_fn,
        M: int,
        sigma: float,
    ):
        print(f"  [Server] Full profiling for {self.num_clients} clients ...")
        for i, client in enumerate(self.clients):
            state = model_state_fn(i)
            rng_i = np.random.default_rng(self.seed + 9000 + i)
            profile = client.compute_loss_profile(
                state, self.model_fn, M, sigma, rng_i,
            )
            self.client_profiles[i] = profile

            if (i + 1) % 20 == 0:
                print(f"    profiled {i + 1}/{self.num_clients}")

        # ── 诊断输出 ──
        self._print_profile_diagnostics()

    def _print_profile_diagnostics(self):
        """打印 loss profile 统计信息，用于验证 probe 是否产生了有效信号。"""
        Z = self._get_profile_matrix()
        print(f"\n  [Diagnostics] Profile matrix shape: {Z.shape}")
        print(f"  [Diagnostics] Per-dimension stats (across clients):")
        print(f"    mean:  {Z.mean(axis=0)[:5].round(3)} ...")
        print(f"    std:   {Z.std(axis=0)[:5].round(3)} ...")
        print(f"    range: {(Z.max(axis=0) - Z.min(axis=0))[:5].round(3)} ...")

        # 检查维度间方差 vs 噪声
        inter_client_std = Z.std(axis=0).mean()
        intra_client_std = Z.std(axis=1).mean()
        print(f"  [Diagnostics] Avg inter-client std (across dims): {inter_client_std:.4f}")
        print(f"  [Diagnostics] Avg intra-client std (across probes): {intra_client_std:.4f}")

        if inter_client_std < 0.01:
            print("  ⚠️  WARNING: Inter-client std very low → probes may lack diversity!")
        else:
            print("  ✅  Profile diversity looks reasonable")

    # ══════════════════════════════════════════════════════
    #  Profile normalization
    # ══════════════════════════════════════════════════════

    def _get_profile_matrix(self) -> np.ndarray:
        M = len(next(iter(self.client_profiles.values())))
        Z = np.zeros((self.num_clients, M), dtype=np.float32)
        for i, prof in self.client_profiles.items():
            Z[i] = prof
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

        norms = np.linalg.norm(R, axis=1, keepdims=True) + 1e-12
        R *= np.minimum(1.0, clip_norm / norms)

        if noise_sigma > 0:
            R += rng.normal(
                0.0, noise_sigma * clip_norm, size=R.shape
            ).astype(np.float32)

        return R

    # ══════════════════════════════════════════════════════
    #  DPMM clustering
    # ══════════════════════════════════════════════════════

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

    # ══════════════════════════════════════════════════════
    #  小组合并
    # ══════════════════════════════════════════════════════

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
                g for g, m in cluster_map.items() if len(m) < min_size
            ]
            big_groups = [
                g for g, m in cluster_map.items() if len(m) >= min_size
            ]

            if not big_groups and small_groups:
                biggest = max(small_groups, key=lambda g: len(cluster_map[g]))
                big_groups = [biggest]
                small_groups = [g for g in small_groups if g != biggest]

            for sg in small_groups:
                if not big_groups:
                    break
                sg_center = centers.get(sg, profiles[cluster_map[sg]].mean(axis=0))
                dists = {
                    bg: np.linalg.norm(sg_center - centers.get(
                        bg, profiles[cluster_map[bg]].mean(axis=0)
                    ))
                    for bg in big_groups
                }
                nearest = min(dists, key=dists.get)

                for i in cluster_map[sg]:
                    self.assignments[i] = nearest
                changed = True

                new_members = cluster_map[nearest] + cluster_map[sg]
                centers[nearest] = profiles[new_members].mean(axis=0)

            if changed:
                cluster_map = self._get_cluster_map()

        self._relabel_assignments()
        self.k_pred = int(len(np.unique(self.assignments)))

    def _relabel_assignments(self):
        old_labels = sorted(np.unique(self.assignments).tolist())
        mapping = {old: new for new, old in enumerate(old_labels)}
        self.assignments = np.array([mapping[int(a)] for a in self.assignments])

    # ══════════════════════════════════════════════════════
    #  Cluster map helper
    # ══════════════════════════════════════════════════════

    def _get_cluster_map(self) -> Dict[int, List[int]]:
        mapping: Dict[int, List[int]] = {}
        for i, c in enumerate(self.assignments):
            mapping.setdefault(int(c), []).append(i)
        return mapping

    # ══════════════════════════════════════════════════════
    #  Group model initialization / re-initialization
    # ══════════════════════════════════════════════════════
    def _cluster_quality_metrics(self) -> Dict[str, float]:
        """
        Compute clustering quality against ground-truth groups.

        These metrics are only for synthetic/controlled experiments where
        client.group_id is available. They are not used by CARES itself.
        """
        y_true = np.array(
            [int(self.clients[i].group_id) for i in range(self.num_clients)],
            dtype=int,
        )
        y_pred = self.assignments.astype(int)

        ari = adjusted_rand_score(y_true, y_pred)
        nmi = normalized_mutual_info_score(y_true, y_pred)

        purity_correct = 0
        for _, members in self._get_cluster_map().items():
            labels = y_true[members]
            if len(labels) == 0:
                continue
            _, counts = np.unique(labels, return_counts=True)
            purity_correct += int(counts.max())

        purity = purity_correct / max(self.num_clients, 1)

        return {
            "ari": float(ari),
            "nmi": float(nmi),
            "purity": float(purity),
        }

    def _init_group_models(self):
        cluster_map = self._get_cluster_map()
        self.cluster_models = {}
        for gid in sorted(cluster_map):
            model = self.model_fn()
            model.load_state_dict(
                copy.deepcopy(self.global_model.state_dict())
            )
            self.cluster_models[gid] = model

        print(f"  [Server] Initialized {len(self.cluster_models)} group models "
              f"from global model")

    def _reinit_group_models(self, old_assignments: np.ndarray):
        old_models = copy.deepcopy(self.cluster_models)
        new_cluster_map = self._get_cluster_map()
        new_models: Dict[int, nn.Module] = {}

        for new_gid, member_ids in new_cluster_map.items():
            old_group_counts: Dict[int, int] = {}
            for mid in member_ids:
                old_g = int(old_assignments[mid])
                old_group_counts[old_g] = old_group_counts.get(old_g, 0) + 1

            states, weights = [], []
            for old_g, count in old_group_counts.items():
                if old_g in old_models:
                    states.append(old_models[old_g].state_dict())
                    weights.append(count)

            if states:
                avg_state = self._weighted_avg_states(states, weights)
            else:
                fallback = next(iter(old_models.values()))
                avg_state = copy.deepcopy(fallback.state_dict())

            model = self.model_fn()
            model.load_state_dict(avg_state)
            new_models[new_gid] = model

        self.cluster_models = new_models
        print(f" [Server] Matched and inherited {len(self.cluster_models)} group models "
              f"after re-clustering")

    @staticmethod
    def _weighted_avg_states(
        states: List[Dict[str, torch.Tensor]],
        weights: List[int],
    ) -> Dict[str, torch.Tensor]:
        total = float(sum(weights))
        avg = copy.deepcopy(states[0])
        for key in avg:
            avg[key] = avg[key].float() * (weights[0] / total)
        for s, w in zip(states[1:], weights[1:]):
            for key in avg:
                avg[key] += s[key].float() * (w / total)
        return avg

    # ══════════════════════════════════════════════════════
    #  Evaluation
    # ══════════════════════════════════════════════════════
    def evaluate_global_model(self) -> Dict[str, float]:
        """Evaluate the warm-up global model before clustering."""
        accs, f1s = [], []
        all_yt, all_yp = [], []
        total_correct, total_n = 0, 0

        for client in self.clients:
            yt, yp = client.evaluate(self.global_model)
            correct = int((yt == yp).sum())
            n = len(yt)

            accs.append(correct / max(n, 1))
            f1s.append(f1_score(yt, yp, average="macro", zero_division=0))

            all_yt.extend(yt.tolist())
            all_yp.extend(yp.tolist())
            total_correct += correct
            total_n += n

        return {
            "client_avg_acc": float(np.mean(accs)),
            "micro_acc": float(total_correct / max(total_n, 1)),
            "client_avg_macro_f1": float(np.mean(f1s)),
            "global_macro_f1": float(f1_score(
                all_yt,
                all_yp,
                average="macro",
                labels=list(range(10)),
                zero_division=0,
            )),
        }

    def evaluate(self) -> Dict[str, float]:
        accs, f1s = [], []
        all_yt, all_yp = [], []
        total_correct, total_n = 0, 0

        for i, client in enumerate(self.clients):
            gid = int(self.assignments[i])
            model = self.cluster_models[gid]
            yt, yp = client.evaluate(model)

            correct = int((yt == yp).sum())
            n = len(yt)

            accs.append(correct / max(n, 1))
            f1s.append(f1_score(yt, yp, average="macro", zero_division=0))

            all_yt.extend(yt.tolist())
            all_yp.extend(yp.tolist())
            total_correct += correct
            total_n += n

        metrics = {
            "k_pred": self.k_pred,
            "client_avg_acc": float(np.mean(accs)),
            "micro_acc": float(total_correct / max(total_n, 1)),
            "client_avg_macro_f1": float(np.mean(f1s)),
            "global_macro_f1": float(f1_score(
                all_yt,
                all_yp,
                average="macro",
                labels=list(range(10)),
                zero_division=0,
            )),
        }

        metrics.update(self._cluster_quality_metrics())
        return metrics

    def _print_eval_snapshot(self, tag: str):
        """Print a lightweight evaluation snapshot during training."""
        metrics = self.evaluate()

        print(
            f"  [Eval:{tag}] "
            f"K={metrics['k_pred']} | "
            f"MicroAcc={metrics['micro_acc']:.4f} | "
            f"ClientAvgAcc={metrics['client_avg_acc']:.4f} | "
            f"GlobalMacroF1={metrics['global_macro_f1']:.4f} | "
            f"ClientAvgMacroF1={metrics['client_avg_macro_f1']:.4f} | "
            f"ARI={metrics.get('ari', float('nan')):.4f} | "
            f"NMI={metrics.get('nmi', float('nan')):.4f} | "
            f"Purity={metrics.get('purity', float('nan')):.4f}"
        )

    # ══════════════════════════════════════════════════════
    #  Print helpers
    # ══════════════════════════════════════════════════════

    def _print_cluster_info(self):
        print(f"  [Server] K = {self.k_pred}")
        for c, members in sorted(self._get_cluster_map().items()):
            # 显示每个簇里各 ground-truth group 的分布
            group_counts: Dict[int, int] = {}
            for i in members:
                g = self.clients[i].group_id
                group_counts[g] = group_counts.get(g, 0) + 1
            group_str = ", ".join(
                f"G{g}:{n}" for g, n in sorted(group_counts.items())
            )
            print(f"    cluster {c}: {len(members)} clients [{group_str}]")

    # ══════════════════════════════════════════════════════
    #  Main pipeline
    # ══════════════════════════════════════════════════════

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
    ) -> Dict[str, float]:

        # ════════════════════════════════════════════════
        #  Phase 0: 初始化
        # ════════════════════════════════════════════════
        print("\n" + "=" * 55)
        print(" Phase 0: Initialization")
        print("=" * 55)

        set_seed(self.seed)
        self.global_model = self.model_fn()
        print(f"  Global model initialized")
        print(f"  Total rounds: {total_rounds}")
        print(f"  Warm-up rounds: {warmup_rounds}")
        print(f"  Cluster interval τ: {cluster_interval}")
        print(f"  Probe pool M: {probe_pool_size}, σ: {probe_sigma}")
        print(f"  Probes: 1 original + 10 class-ablation + "
              f"{max(0, probe_pool_size - 11)} random")

        # ════════════════════════════════════════════════
        #  Phase 1: Warm-up 全局 FedAvg
        # ════════════════════════════════════════════════
        print("\n" + "=" * 55)
        print(" Phase 1: Warm-up Global FedAvg")
        print("=" * 55)

        for t in range(warmup_rounds):
            selected = self._select_clients(client_frac)
            global_state = copy.deepcopy(self.global_model.state_dict())

            deltas, weights = [], []

            for idx in selected:
                client = self.clients[idx]

                rng_i = np.random.default_rng(
                    self.seed + 8000 * t + idx
                )
                profile = client.compute_loss_profile(
                    global_state, self.model_fn,
                    probe_pool_size, probe_sigma, rng_i,
                )
                self.client_profiles[idx] = profile

                delta = client.local_train(
                    global_state, self.model_fn, local_epochs, lr,
                )
                deltas.append(delta)
                weights.append(client.num_train)

            avg_delta = self._aggregate_deltas(deltas, weights)
            new_state = self._apply_delta(
                self.global_model.state_dict(), avg_delta,
            )
            self.global_model.load_state_dict(new_state)

            print(f"  [Warmup] Round {t + 1}/{warmup_rounds} done "
                  f"(selected {len(selected)} clients)")

            warm_metrics = self.evaluate_global_model()
            print(
                f"  [Eval:Warmup R{t + 1}] "
                f"MicroAcc={warm_metrics['micro_acc']:.4f} | "
                f"ClientAvgAcc={warm_metrics['client_avg_acc']:.4f} | "
                f"GlobalMacroF1={warm_metrics['global_macro_f1']:.4f}"
            )

        # ════════════════════════════════════════════════
        #  过渡：首次全量 Profiling + DPMM 聚类
        # ════════════════════════════════════════════════
        print("\n" + "=" * 55)
        print(" Transition: First DPMM Clustering")
        print("=" * 55)

        self._full_profiling(
            model_state_fn=lambda _: copy.deepcopy(
                self.global_model.state_dict()
            ),
            M=probe_pool_size,
            sigma=probe_sigma,
        )

        Z = self._get_profile_matrix()
        profiles_normed = self._normalize_profiles(Z, clip_norm, noise_sigma)
        self._run_dpmm(profiles_normed, dpmm_max_comp, dpmm_prior)
        self._merge_small_clusters(min_cluster_size, profiles_normed)
        self._print_cluster_info()
        self._init_group_models()

        # ════════════════════════════════════════════════
        #  Phase 2: 分组 FedAvg + 动态重聚类
        # ════════════════════════════════════════════════
        print("\n" + "=" * 55)
        print(" Phase 2: Clustered FedAvg with Dynamic Re-clustering")
        print("=" * 55)

        clustered_rounds = total_rounds - warmup_rounds

        for t_c in range(clustered_rounds):
            t_global = warmup_rounds + t_c

            # ── 是否需要重聚类？ ──
            if t_c > 0 and t_c % cluster_interval == 0:
                print(f"\n  [Server] Re-clustering at round {t_global + 1} ...")

                old_assignments = self.assignments.copy()

                self._full_profiling(
                    model_state_fn=lambda i: copy.deepcopy(
                        self.cluster_models[
                            int(self.assignments[i])
                        ].state_dict()
                    ),
                    M=probe_pool_size,
                    sigma=probe_sigma,
                )

                Z = self._get_profile_matrix()
                profiles_normed = self._normalize_profiles(
                    Z, clip_norm, noise_sigma,
                )
                self._run_dpmm(profiles_normed, dpmm_max_comp, dpmm_prior)
                self._merge_small_clusters(min_cluster_size, profiles_normed)
                self._print_cluster_info()
                self._reinit_group_models(old_assignments)

            # ── 选择本轮参与的 client ──
            selected = self._select_clients(client_frac)

            # ── 按组收集 delta ──
            group_deltas: Dict[int, Tuple[List, List]] = {
                g: ([], []) for g in self.cluster_models
            }

            for idx in selected:
                client = self.clients[idx]
                gid = int(self.assignments[idx])
                group_state = copy.deepcopy(
                    self.cluster_models[gid].state_dict()
                )

                rng_i = np.random.default_rng(
                    self.seed + 8000 * t_global + idx
                )
                profile = client.compute_loss_profile(
                    group_state, self.model_fn,
                    probe_pool_size, probe_sigma, rng_i,
                )
                self.client_profiles[idx] = profile

                delta = client.local_train(
                    group_state, self.model_fn, local_epochs, lr,
                )
                group_deltas[gid][0].append(delta)
                group_deltas[gid][1].append(client.num_train)

            # ── Server: 分组聚合 ──
            for gid in self.cluster_models:
                g_deltas, g_weights = group_deltas[gid]
                if g_deltas:
                    avg_delta = self._aggregate_deltas(g_deltas, g_weights)
                    new_state = self._apply_delta(
                        self.cluster_models[gid].state_dict(), avg_delta,
                    )
                    self.cluster_models[gid].load_state_dict(new_state)

            if (t_c + 1) % 5 == 0 or t_c == 0:
                print(f"  [Clustered] Round {t_global + 1}/{total_rounds} done "
                      f"(selected {len(selected)} clients)")
                self._print_eval_snapshot(f"R{t_global + 1}")

        # ════════════════════════════════════════════════
        #  Evaluation
        # ════════════════════════════════════════════════
        print("\n" + "=" * 55)
        print(" Evaluation")
        print("=" * 55)

        metrics = self.evaluate()
        return metrics