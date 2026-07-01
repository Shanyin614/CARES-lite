"""Federated Client for CARES-Lite."""

import copy
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from src.data import ClientMeta
from src.model import get_last_linear_keys


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
        num_classes: int,
    ):
        self.id = meta.client_id
        self.group_id = meta.group_id
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = num_classes

        self.train_set = Subset(train_dataset, meta.train_indices)
        self.val_set = Subset(train_dataset, meta.val_indices)
        self.test_set = Subset(test_dataset, meta.test_indices)
        self.num_train = len(meta.train_indices)

    @staticmethod
    def _state_to_cpu(
        state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Return a detached CPU copy of a state dict."""
        return {
            key: value.detach().cpu().clone()
            for key, value in state.items()
        }

    @staticmethod
    def _resolve_probe_seed(seed_like: Any) -> int:
        """Resolve the seed used by random probes.

        Backward compatibility:
        - If server passes an int, use it as probe_seed.
        - If server still passes np.random.Generator, use a fixed seed so that
          all clients share the same perturbation directions for each probe.
        """
        if isinstance(seed_like, (int, np.integer)):
            return int(seed_like)
        return 0

    @staticmethod
    def _build_probe_pool(
        base_state: Dict[str, torch.Tensor],
        M: int,
        sigma: float,
        rng: Any,
        num_classes: int,
    ) -> List[Dict[str, torch.Tensor]]:
        """Construct M probe states.

        Probe 0 is the original model. The next probes are class-ablation probes
        on the final linear layer. Remaining probes add deterministic Gaussian
        perturbations to floating-point parameters/buffers.
        """
        base_state = FLClient._state_to_cpu(base_state)

        probes: List[Dict[str, torch.Tensor]] = []
        probe_seed = FLClient._resolve_probe_seed(rng)

        probes.append(copy.deepcopy(base_state))

        n_ablation = min(M - 1, num_classes)
        weight_key, bias_key = get_last_linear_keys(base_state)

        for c in range(n_ablation):
            perturbed = copy.deepcopy(base_state)

            if c < perturbed[weight_key].shape[0]:
                perturbed[weight_key][c, :] = 0.0

            if bias_key is not None and c < perturbed[bias_key].shape[0]:
                perturbed[bias_key][c] = -100.0

            probes.append(perturbed)

        n_random = M - 1 - n_ablation

        for r in range(n_random):
            perturbed = copy.deepcopy(base_state)

            for key_idx, key in enumerate(perturbed):
                tensor = perturbed[key]

                if not torch.is_floating_point(tensor):
                    continue

                gen = torch.Generator(device="cpu")
                gen.manual_seed(probe_seed + 1009 * r + 9176 * key_idx)

                noise = torch.randn(
                    tensor.shape,
                    generator=gen,
                    dtype=torch.float32,
                )

                noise = noise.to(
                    device=tensor.device,
                    dtype=tensor.dtype,
                ) * sigma

                perturbed[key] = tensor + noise

            probes.append(perturbed)

        return probes

    @torch.no_grad()
    def compute_loss_profile(
        self,
        base_state: Dict[str, torch.Tensor],
        model_cls: type,
        M: int,
        sigma: float,
        rng: Any,
    ) -> np.ndarray:
        """Compute a local validation loss vector over the probe pool."""
        base_state = self._state_to_cpu(base_state)

        probe_states = self._build_probe_pool(
            base_state,
            M,
            sigma,
            rng,
            self.num_classes,
        )

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
            model.to(self.device)
            model.eval()

            total_loss = 0.0
            total_n = 0

            for x, y in val_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = model(x)

                total_loss += F.cross_entropy(
                    logits,
                    y,
                    reduction="sum",
                ).item()

                total_n += y.numel()

            profile[h] = total_loss / max(total_n, 1)

            model.cpu()
            del model

        return profile

    def local_train(
        self,
        global_state: Dict[str, torch.Tensor],
        model_cls: type,
        local_epochs: int,
        lr: float,
    ) -> Dict[str, torch.Tensor]:
        """Run local SGD and return CPU delta weights."""
        original_state = self._state_to_cpu(global_state)

        model: nn.Module = model_cls()
        model.load_state_dict(copy.deepcopy(original_state))
        model.to(self.device)
        model.train()

        loader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"),
        )

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
        )

        for _ in range(local_epochs):
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad(set_to_none=True)

                logits = model(x)
                loss = F.cross_entropy(logits, y)

                loss.backward()
                optimizer.step()

        updated_state = self._state_to_cpu(model.state_dict())

        model.cpu()
        del model

        delta: Dict[str, torch.Tensor] = {}

        for key in updated_state:
            if torch.is_floating_point(updated_state[key]):
                delta[key] = (
                    updated_state[key].float()
                    - original_state[key].float()
                )
            else:
                # Non-floating buffers are not updated by SGD.
                delta[key] = torch.zeros_like(updated_state[key])

        return delta

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate without moving the server's original model.

        A temporary copy is moved to GPU for local inference. This prevents
        server-side global/group models from being changed from CPU to CUDA.
        """
        eval_model = copy.deepcopy(model)
        eval_model.to(self.device)
        eval_model.eval()

        loader = DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"),
        )

        all_y_true = []
        all_y_pred = []

        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)

            logits = eval_model(x)
            pred = logits.argmax(dim=1)

            all_y_true.append(y.detach().cpu())
            all_y_pred.append(pred.detach().cpu())

        eval_model.cpu()
        del eval_model

        if not all_y_true:
            return (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
            )

        y_true = torch.cat(all_y_true).numpy()
        y_pred = torch.cat(all_y_pred).numpy()

        return y_true, y_pred
