import torch
import heapq
import os
import logging
from typing import List, Set, Tuple
from .abstract import AbstractEplbPolicy

logger = logging.getLogger(__name__)

class OmniEplbPolicy(AbstractEplbPolicy):
    """
    Enhanced OmniPlacement Policy for vLLM.
    
    Algorithm derived from Huawei OmniInfer open source project: 
    https://github.com/omni-ai-npu/omni-infer
    
    Features:
    1. Allocation: Priority Queue based on load (supports Log Normalization).
    2. Distribution: Greedy Min-Heap for load balancing.
    3. Topology: Host-Aware Reordering.
    """

    # --- Feature Switches (Defaults loaded from Env Vars) ---
    
    # Enable log1p(load) for replica allocation to prevent hot experts 
    # from monopolizing redundant slots. Default: False.
    ENABLE_LOG_NORMALIZATION = os.getenv("VLLM_OMNI_LOG_NORM", "0") == "1"
    
    # Enable round-robin rank shuffling across hosts to distribute 
    # hot spots and balance node-level bandwidth. Default: True.
    ENABLE_HOST_REORDERING = os.getenv("VLLM_OMNI_HOST_REORDER", "1") == "1"

    @classmethod
    def rebalance_experts(
        cls,
        weight: torch.Tensor,  # [num_layers, num_experts]
        num_replicas: int,     # Total physical slots
        num_groups: int,
        num_nodes: int,        # Corresponds to num_hosts
        num_ranks: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        num_layers, num_experts = weight.shape
        device = weight.device
        slots_per_rank = num_replicas // num_ranks
        
        # Prepare load data
        # Raw loads are used for physical distribution to ensure real load balancing.
        weight_cpu_float = weight.float().cpu()
        loads_raw_list = weight_cpu_float.tolist()

        # Allocation loads: determine if log normalization is applied
        if cls.ENABLE_LOG_NORMALIZATION:
            loads_alloc_list = torch.log1p(weight_cpu_float).tolist()
        else:
            loads_alloc_list = loads_raw_list

        phy2log_list = []

        for layer_idx in range(num_layers):
            # 1. Replica Allocation
            # Use loads_alloc_list (potentially log-normalized)
            deployments = cls._allocate_expert_deployments(
                loads_alloc_list[layer_idx], 
                num_experts, 
                num_replicas
            )
            
            # 2. Load Distribution
            # Always use raw loads to balance actual computation
            placement_matrix = cls._distribute_experts_to_ranks(
                loads_raw_list[layer_idx],
                deployments,
                num_ranks,
                slots_per_rank,
                num_experts
            )
            
            # 3. Host-Aware Reordering
            # Only execute if enabled and there are multiple nodes
            if cls.ENABLE_HOST_REORDERING and num_nodes > 1:
                placement_matrix = cls._reorder_for_hosts(
                    placement_matrix, 
                    num_ranks, 
                    num_nodes
                )
            
            # Flatten the placement matrix
            flat_layer_placement = []
            for r in range(num_ranks):
                flat_layer_placement.extend(placement_matrix[r])
            
            phy2log_list.append(flat_layer_placement)

        # --- Convert to vLLM Tensor format ---
        phy2log = torch.tensor(phy2log_list, dtype=torch.int64, device=device)
        
        # Calculate logcnt (replica count per logical expert)
        logcnt = torch.zeros((num_layers, num_experts), dtype=torch.int64, device=device)
        for i in range(num_experts):
            logcnt[:, i] = (phy2log == i).sum(dim=1)
            
        # Calculate log2phy (logical to physical mapping)
        max_replicas_count = int(logcnt.max().item())
        log2phy = torch.full(
            (num_layers, num_experts, max_replicas_count),
            -1,
            dtype=torch.int64,
            device=device
        )
        
        phy2log_cpu = phy2log.cpu().numpy()
        log2phy_cpu = log2phy.cpu().numpy()
        
        # Fill log2phy (CPU loop)
        for l in range(num_layers):
            counts = {}
            for p_idx, expert_id in enumerate(phy2log_cpu[l]):
                if expert_id == -1: continue
                c = counts.get(expert_id, 0)
                if c < max_replicas_count:
                    log2phy_cpu[l, expert_id, c] = p_idx
                    counts[expert_id] = c + 1
        
        log2phy.copy_(torch.from_numpy(log2phy_cpu))

        return phy2log, log2phy, logcnt

    @staticmethod
    def _allocate_expert_deployments(loads: List[float], num_experts: int, total_slots: int) -> List[int]:
        """Use max-heap for replica allocation."""
        deployments = [1] * num_experts
        remaining_budget = total_slots - num_experts
        
        if remaining_budget <= 0:
            return deployments

        pq = []
        for i in range(num_experts):
            # Python heapq is min-heap, store negative values for max-heap behavior
            priority = loads[i] 
            if priority > 0:
                heapq.heappush(pq, (-priority, i))

        while remaining_budget > 0 and pq:
            neg_priority, expert_id = heapq.heappop(pq)
            
            deployments[expert_id] += 1
            remaining_budget -= 1
            
            # Update priority: priority = load / current_count
            new_priority = loads[expert_id] / deployments[expert_id]
            heapq.heappush(pq, (-new_priority, expert_id))
            
        return deployments

    @staticmethod
    def _distribute_experts_to_ranks(
        loads: List[float],
        deployments: List[int],
        num_ranks: int,
        slots_per_rank: int,
        num_experts: int
    ) -> List[List[int]]:
        """Greedy load balancing using min-heap."""
        instances = []
        for expert_id in range(num_experts):
            count = deployments[expert_id]
            if count > 0:
                load_share = loads[expert_id] / count
                for _ in range(count):
                    instances.append((load_share, expert_id))
        
        # Sort by load descending (Longest Processing Time first)
        instances.sort(key=lambda x: x[0], reverse=True)
        
        # Min-heap for rank loads: (current_load, rank_id)
        rank_heap = [] 
        for r in range(num_ranks):
            heapq.heappush(rank_heap, (0.0, r))
            
        placement_matrix = [[-1] * slots_per_rank for _ in range(num_ranks)]
        rank_filled_slots = [0] * num_ranks
        rank_contents: List[Set[int]] = [set() for _ in range(num_ranks)]
        
        for load_share, expert_id in instances:
            popped = []
            selected_rank = -1
            
            # Try to find the least loaded rank that satisfies constraints
            while rank_heap:
                curr_load, r = heapq.heappop(rank_heap)
                
                # Constraints: Rank not full AND expert not already present
                if rank_filled_slots[r] < slots_per_rank and expert_id not in rank_contents[r]:
                    selected_rank = r
                    
                    placement_matrix[r][rank_filled_slots[r]] = expert_id
                    rank_filled_slots[r] += 1
                    rank_contents[r].add(expert_id)
                    
                    new_load = curr_load + load_share
                    heapq.heappush(rank_heap, (new_load, r))
                    break
                else:
                    popped.append((curr_load, r))
            
            # Restore heap
            for item in popped:
                heapq.heappush(rank_heap, item)
                
            # Fallback: Force assignment to least loaded rank (ignoring duplication constraint)
            if selected_rank == -1:
                popped = []
                while rank_heap:
                    curr_load, r = heapq.heappop(rank_heap)
                    if rank_filled_slots[r] < slots_per_rank:
                        placement_matrix[r][rank_filled_slots[r]] = expert_id
                        rank_filled_slots[r] += 1
                        heapq.heappush(rank_heap, (curr_load + load_share, r))
                        break
                    popped.append((curr_load, r))
                for item in popped:
                    heapq.heappush(rank_heap, item)

        return placement_matrix

    @staticmethod
    def _reorder_for_hosts(
        placement_matrix: List[List[int]], 
        num_ranks: int, 
        num_nodes: int
    ) -> List[List[int]]:
        """
        Host-Aware Reordering: Map ranks to hosts in a round-robin fashion.
        """
        ranks_per_node = num_ranks // num_nodes
        new_matrix = [None] * num_ranks
        
        # Track current rank index per node
        node_cur_rank_idx = [0] * num_nodes
        
        for i in range(num_ranks):
            # Original logical rank i maps to target node (i % num_nodes)
            target_node = i % num_nodes
            
            # Calculate physical Rank ID
            physical_rank = target_node * ranks_per_node + node_cur_rank_idx[target_node]
            
            if physical_rank < num_ranks:
                new_matrix[physical_rank] = placement_matrix[i]
                node_cur_rank_idx[target_node] += 1
            else:
                new_matrix[i] = placement_matrix[i] 

        return new_matrix