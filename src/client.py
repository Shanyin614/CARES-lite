"""
Federated Client (CARES-Lite)
==============================
每个 Client 暴露三个接口给 Server：

  1. compute_loss_profile()  — 基于当前模型构造 probe pool，在 val set 上计算 loss 向量
  2. local_train()           — 本地 SGD，返回模型增量 Δw
  3. evaluate()              — 在本地 test set 上推理
"""

import copy
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from src.data import ClientMeta
from src.model import LAST_LAYER_PREFIX


NUM_CLASSES = 10


class FLClient:
    """Represents a single federated learning participant."""

    def __init__(
        self,
        meta: ClientMeta,
        train_dataset: Dataset,
        test_dataset: Dataset,
        batch_size: int,
        num_workers: int,
        device: torch.device,
    ):
        self.id = meta.client_id
        self.group_id = meta.group_id
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_set = Subset(train_dataset, meta.train_indices)
        self.val_set = Subset(train_dataset, meta.val_indices)
        self.test_set = Subset(test_dataset, meta.test_indices)
        self.num_train = len(meta.train_indices)

    # ─────────────────────────────────────────────────────
    #  Probe Pool 构造（Class-Ablation + Random Perturbation）
    # ─────────────────────────────────────────────────────

    @staticmethod
    def _build_probe_pool(
        base_state: Dict[str, torch.Tensor],
        M: int,
        sigma: float,
        rng: np.random.Generator,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        构造 M 个 probe 模型的 state_dict：

        Probe 0         : 原始模型 W(t)
        Probe 1 ~ 10    : Class-Ablation —— 屏蔽 class c 的输出层权重
                          weight[c, :] = 0, bias[c] = -100
                          → 使 probe 对 class c "失明"
                          → 数据富含 class c 的 client loss 显著升高
        Probe 11 ~ M-1  : 对所有层添加较大随机噪声 N(0, σ²)
                          → 提供额外非结构化多样性

        这种结构化 probe 使得 loss profile 能反映 client 的 label 分布差异，
        从而让 DPMM 可以准确区分不同数据分布的 client。
        """
        probes: List[Dict[str, torch.Tensor]] = []

        # ── Probe 0: 原始模型 ──
        probes.append(base_state)

        # ── Probe 1 ~ min(M-1, NUM_CLASSES): Class-Ablation ──
        n_ablation = min(M - 1, NUM_CLASSES)
        weight_key = LAST_LAYER_PREFIX + ".weight"
        bias_key = LAST_LAYER_PREFIX + ".bias"

        for c in range(n_ablation):
            perturbed = copy.deepcopy(base_state)
            if weight_key in perturbed:
                perturbed[weight_key][c, :] = 0.0
            if bias_key in perturbed:
                perturbed[bias_key][c] = -100.0
            probes.append(perturbed)

        # ── Probe (n_ablation+1) ~ M-1: 全层随机扰动 ──
        n_random = M - 1 - n_ablation
        for _ in range(n_random):
            perturbed = copy.deepcopy(base_state)
            for key in perturbed:
                noise = torch.randn_like(perturbed[key]) * sigma
                perturbed[key] = perturbed[key] + noise
            probes.append(perturbed)

        return probes

    # ─────────────────────────────────────────────────────
    #  Loss profiling
    # ─────────────────────────────────────────────────────

    @torch.no_grad()
    def compute_loss_profile(
        self,
        base_state: Dict[str, torch.Tensor],
        model_cls: type,
        M: int,
        sigma: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Server 下发当前模型参数 base_state →
        Client 本地构造 M 个 probe (class-ablation + random) →
        在本地 val set 上计算 M 维 loss 向量 →
        上传该向量。

        Returns:
            profile: shape (M,), 每个元素 = 该 probe 在 val set 上的 avg CE loss
        """
        probe_states = self._build_probe_pool(base_state, M, sigma, rng)
        profile = np.zeros(M, dtype=np.float32)

        val_loader = DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"),
        )

        for h, state in enumerate(probe_states):
            model: nn.Module = model_cls()
            model.load_state_dict(state)
            model.to(self.device).eval()

            total_loss, total_n = 0.0, 0
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                total_loss += F.cross_entropy(
                    model(x), y, reduction="sum"
                ).item()
                total_n += y.numel()

            profile[h] = total_loss / max(total_n, 1)
            model.cpu()

        return profile

    # ─────────────────────────────────────────────────────
    #  Local training → 返回 Δw
    # ─────────────────────────────────────────────────────

    def local_train(
        self,
        global_state: Dict[str, torch.Tensor],
        model_cls: type,
        local_epochs: int,
        lr: float,
    ) -> Dict[str, torch.Tensor]:
        """
        接收 Server 下发的全局/组模型参数 →
        本地 SGD →
        返回模型增量 Δw = W̃ - W。
        """
        original_state = copy.deepcopy(global_state)

        model: nn.Module = model_cls()
        model.load_state_dict(copy.deepcopy(global_state))
        model.to(self.device).train()

        loader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"),
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

        for _ in range(local_epochs):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                F.cross_entropy(model(x), y).backward()
                optimizer.step()

        updated_state = model.cpu().state_dict()
        delta: Dict[str, torch.Tensor] = {}
        for key in updated_state:
            delta[key] = updated_state[key].float() - original_state[key].float()

        return delta

    # ─────────────────────────────────────────────────────
    #  Evaluation
    # ─────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, model: nn.Module) -> Tuple[np.ndarray, np.ndarray]:
        """在本地测试集上推理，返回 (y_true, y_pred)。"""
        model.to(self.device).eval()

        loader = DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"),
        )

        yt, yp = [], []
        for x, y in loader:
            preds = model(x.to(self.device)).argmax(1).cpu().numpy()
            yt.extend(y.numpy().tolist())
            yp.extend(preds.tolist())

        model.cpu()
        return np.array(yt), np.array(yp)
