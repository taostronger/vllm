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

        # Aggregate across all layers to find a shared lambda.
        agg_load = weight.float().sum(dim=0)
        agg_aff = affinity_matrix.float().sum(dim=0)
        _, best_lam = cls._find_optimal_params(
            agg_load, agg_aff, num_ranks, slots_per_rank,
            num_logical_experts, num_redundant,
        )
        best_beta = 1.0 - best_lam

        all_phy2log = torch.zeros((num_layers, num_replicas),
                                  dtype=torch.int64, device=device)
        all_log2phy = torch.full(
            (num_layers, num_logical_experts, num_redundant + 1), -1,
            dtype=torch.int64, device=device,
        )
        all_logcnt = torch.ones((num_layers, num_logical_experts),
                                dtype=torch.int64, device=device)

        for layer_idx in range(num_layers):
            load = weight[layer_idx].float()
            aff = affinity_matrix[layer_idx].float()

            placement = cls._placement_normalized(
                load, aff, num_ranks, slots_per_rank, best_lam, best_beta,
            )

            phy2log = torch.zeros(num_replicas, dtype=torch.int64,
                                  device=device)
            for r in range(num_ranks):
                start = r * slots_per_rank
                end = start + slots_per_rank
                phy2log[start:end] = placement[r]

            log2phy = torch.full(
                (num_logical_experts, num_redundant + 1), -1,
                dtype=torch.int64, device=device,
            )
            for r in range(num_ranks):
                start = r * slots_per_rank
                for i, e in enumerate(placement[r]):
                    log2phy[e, 0] = start + i

            all_phy2log[layer_idx] = phy2log
            all_log2phy[layer_idx] = log2phy

        if old_global_expert_indices is not None:
            all_phy2log_np = all_phy2log.cpu().numpy()
            old_np = old_global_expert_indices.cpu().numpy()
            all_phy2log_np, _ = DefaultEplbPolicy.preserve_intragpu_slots(
                all_phy2log_np,
                np.zeros_like(all_phy2log_np),
                num_ranks, old_np,
            )
            all_phy2log = torch.from_numpy(all_phy2log_np).to(device)

        return (all_phy2log, all_log2phy, all_logcnt)

    @classmethod
    def _find_optimal_params(
        cls,
        load: torch.Tensor,
        affinity: torch.Tensor,
        num_ranks: int,
        slots_per_rank: int,
        num_experts: int,
        num_redundant: int,
    ) -> tuple[list[torch.Tensor], float]:
        """Hybrid coarse-to-fine parameter search (golden section)."""
        coarse_points = [0.0, 0.5, 1.0]
        best_lam = 0.5
        best_score = float('inf')

        for lam in coarse_points:
            beta = 1.0 - lam
            placement = cls._placement_normalized(
                load, affinity, num_ranks, slots_per_rank, lam, beta,
            )
            score = cls._evaluate_latency(placement, load, affinity,
                                          num_ranks)
            if score < best_score:
                best_score = score
                best_lam = lam

        left = max(0.0, best_lam - 0.25)
        right = min(1.0, best_lam + 0.25)
        phi = 0.6180339887

        for _ in range(5):
            if right - left < 0.001:
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
        load: torch.Tensor,
        affinity: torch.Tensor,
        num_ranks: int,
        slots_per_rank: int,
        lam: float,
        beta: float,
    ) -> list[torch.Tensor]:
        """
        Normalized greedy placement (torch, on-device).

        Score(e, r) = lam * NormAff(e, r) - beta * NormLoad(r)
        """
        num_experts = load.shape[0]
        placement: list[list[int]] = [[] for _ in range(num_ranks)]
        rank_loads = torch.zeros(num_ranks, dtype=torch.float32,
                                 device=load.device)
        total_load = load.sum().item()
        avg_load = total_load / num_ranks if num_ranks > 0 else 1.0

        sorted_experts = torch.argsort(-load)

        for e_idx in range(num_experts):
            e = sorted_experts[e_idx].item()
            load_e = load[e].item()
            scores = torch.zeros(num_ranks, dtype=torch.float32,
                                 device=load.device)
            for r in range(num_ranks):
                norm_aff = affinity[e, r].item() / max(load_e, 1.0)
                norm_load = rank_loads[r].item() / max(avg_load, 1e-5)
                scores[r] = lam * norm_aff - beta * norm_load

            ranked = torch.argsort(-scores)
            placed = False
            for r_idx in range(num_ranks):
                r = ranked[r_idx].item()
                if len(placement[r]) < slots_per_rank:
                    placement[r].append(e)
                    rank_loads[r] += load_e
                    placed = True
                    break

            if not placed:
                for r in range(num_ranks):
                    if len(placement[r]) < slots_per_rank:
                        placement[r].append(e)
                        rank_loads[r] += load_e
                        break

        return [torch.tensor(p, dtype=torch.int64, device=load.device)
                for p in placement]

    @classmethod
    def _evaluate_latency(
        cls,
        placement: list[torch.Tensor],
        load: torch.Tensor,
        affinity: torch.Tensor,
        num_ranks: int,
        t_compute: float = 0.05,
        t_comm: float = 0.30,
    ) -> float:
        """Lightweight latency model for parameter search."""
        rank_loads = [0.0] * num_ranks
        remote_tokens = 0.0

        for r in range(num_ranks):
            for e_tensor in placement[r]:
                e = e_tensor.item() if e_tensor.dim() == 0 else e_tensor
                e = int(e)
                load_e = load[e].item()
                rank_loads[r] += load_e
                local_tokens = affinity[e, r].item()
                total_tokens = affinity[e].sum().item()
                remote_tokens += max(0, total_tokens - local_tokens)

        max_load = max(rank_loads) if rank_loads else 0
        latency = max_load * t_compute + remote_tokens * t_comm / num_ranks
        return latency
