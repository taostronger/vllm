# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Adaptive communication-aware EPLB policy.

Optimizes expert placement to reduce cross-rank AllToAll communication
while maintaining load balance. Uses a normalized greedy placement
algorithm with automatic parameter search (golden section search).
"""

import numpy as np
import torch

from .abstract import AbstractEplbPolicy
from .default import DefaultEplbPolicy


class AdaptiveEplbPolicy(AbstractEplbPolicy):

    @classmethod
    def rebalance_experts(
        cls,
        weight: torch.Tensor,
        num_replicas: int,
        num_groups: int,
        num_nodes: int,
        num_ranks: int,
        old_global_expert_indices: torch.Tensor | None = None,
        affinity_matrix: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Adaptive communication-aware expert placement.

        Falls back to DefaultEplbPolicy when affinity data is unavailable.
        """
        if affinity_matrix is None:
            return DefaultEplbPolicy.rebalance_experts(
                weight, num_replicas, num_groups, num_nodes,
                num_ranks, old_global_expert_indices,
            )

        device = weight.device
        num_layers, num_logical_experts = weight.shape
        num_redundant = num_replicas - num_logical_experts
        slots_per_rank = num_replicas // num_ranks

        weight_np = weight.float().cpu().numpy()
        affinity_np = affinity_matrix.float().cpu().numpy()

        all_phy2log = np.zeros((num_layers, num_replicas), dtype=np.int64)
        all_log2phy = np.full(
            (num_layers, num_logical_experts, num_redundant + 1), -1,
            dtype=np.int64,
        )
        all_logcnt = np.ones((num_layers, num_logical_experts), dtype=np.int64)

        for layer_idx in range(num_layers):
            load = weight_np[layer_idx]
            aff = affinity_np[layer_idx]

            best_placement, best_lam = cls._find_optimal_params(
                load, aff, num_ranks, slots_per_rank,
                num_logical_experts, num_redundant,
            )

            phy2log = np.zeros(num_replicas, dtype=np.int64)
            for r in range(num_ranks):
                start = r * slots_per_rank
                end = start + slots_per_rank
                for i, e in enumerate(best_placement[r]):
                    phy2log[start + i] = e

            logcnt = np.ones(num_logical_experts, dtype=np.int64)
            log2phy = np.full(
                (num_logical_experts, num_redundant + 1), -1, dtype=np.int64,
            )
            for r in range(num_ranks):
                start = r * slots_per_rank
                for i, e in enumerate(best_placement[r]):
                    log2phy[e, 0] = start + i

            all_phy2log[layer_idx] = phy2log
            all_log2phy[layer_idx] = log2phy
            all_logcnt[layer_idx] = logcnt

        if old_global_expert_indices is not None:
            old_np = old_global_expert_indices.cpu().numpy()
            all_phy2log, _ = DefaultEplbPolicy.preserve_intragpu_slots(
                all_phy2log,
                np.zeros_like(all_phy2log),
                num_ranks, old_np,
            )

        return (
            torch.from_numpy(all_phy2log).to(device),
            torch.from_numpy(all_log2phy).to(device),
            torch.from_numpy(all_logcnt).to(device),
        )

    @classmethod
    def _find_optimal_params(
        cls,
        load: np.ndarray,
        affinity: np.ndarray,
        num_ranks: int,
        slots_per_rank: int,
        num_experts: int,
        num_redundant: int,
    ) -> tuple[list[list[int]], float]:
        """Hybrid coarse-to-fine parameter search (golden section)."""
        coarse_points = [0.0, 0.25, 0.5, 0.75, 1.0]
        best_lam = 0.5
        best_score = float('inf')

        for lam in coarse_points:
            beta = 1.0 - lam
            placement = cls._placement_normalized(
                load, affinity, num_ranks, slots_per_rank, lam, beta,
            )
            score = cls._evaluate_latency(placement, load, affinity, num_ranks)
            if score < best_score:
                best_score = score
                best_lam = lam

        left = max(0.0, best_lam - 0.25)
        right = min(1.0, best_lam + 0.25)
        phi = 0.6180339887

        for _ in range(10):
            if right - left < 0.0001:
                break
            w1 = right - phi * (right - left)
            w2 = left + phi * (right - left)
            s1 = cls._evaluate_latency(
                cls._placement_normalized(
                    load, affinity, num_ranks, slots_per_rank, w1, 1 - w1),
                load, affinity, num_ranks,
            )
            s2 = cls._evaluate_latency(
                cls._placement_normalized(
                    load, affinity, num_ranks, slots_per_rank, w2, 1 - w2),
                load, affinity, num_ranks,
            )
            if s1 < s2:
                right = w2
            else:
                left = w1

        best_lam = (left + right) / 2
        best_placement = cls._placement_normalized(
            load, affinity, num_ranks, slots_per_rank, best_lam, 1 - best_lam,
        )
        return best_placement, best_lam

    @classmethod
    def _placement_normalized(
        cls,
        load: np.ndarray,
        affinity: np.ndarray,
        num_ranks: int,
        slots_per_rank: int,
        lam: float,
        beta: float,
    ) -> list[list[int]]:
        """
        Normalized greedy placement.

        Score(e, r) = lam * NormAff(e, r) - beta * NormLoad(r)
        """
        num_experts = len(load)
        placement: list[list[int]] = [[] for _ in range(num_ranks)]
        rank_loads = np.zeros(num_ranks, dtype=np.float64)
        total_load = load.sum()
        avg_load = total_load / num_ranks if num_ranks > 0 else 1.0

        sorted_experts = np.argsort(-load)

        for e in sorted_experts:
            scores = np.zeros(num_ranks, dtype=np.float64)
            for r in range(num_ranks):
                norm_aff = affinity[e, r] / max(load[e], 1)
                norm_load = rank_loads[r] / max(avg_load, 1e-5)
                scores[r] = lam * norm_aff - beta * norm_load

            ranked = np.argsort(-scores)
            placed = False
            for r in ranked:
                if len(placement[r]) < slots_per_rank:
                    placement[r].append(int(e))
                    rank_loads[r] += load[e]
                    placed = True
                    break

            if not placed:
                for r in range(num_ranks):
                    if len(placement[r]) < slots_per_rank:
                        placement[r].append(int(e))
                        rank_loads[r] += load[e]
                        break

        return placement

    @classmethod
    def _evaluate_latency(
        cls,
        placement: list[list[int]],
        load: np.ndarray,
        affinity: np.ndarray,
        num_ranks: int,
        t_compute: float = 0.05,
        t_comm: float = 0.30,
    ) -> float:
        """Lightweight latency model for parameter search."""
        rank_loads = np.zeros(num_ranks, dtype=np.float64)
        remote_tokens = 0.0

        for r in range(num_ranks):
            for e in placement[r]:
                rank_loads[r] += load[e]
                local_tokens = affinity[e, r]
                total_tokens = affinity[e].sum()
                remote_tokens += max(0, total_tokens - local_tokens)

        max_load = max(rank_loads) if len(rank_loads) > 0 else 0
        latency = max_load * t_compute + remote_tokens * t_comm / num_ranks
        return latency
