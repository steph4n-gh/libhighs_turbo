import time
import numpy as np
import torch

from typing import Any, Dict, List, Tuple

from highs_turbo.api import TurboSolver
from highs_turbo.graph_generator import GraphInstance
from highs_turbo.compiled_engine import CompiledBitGraph, solve_milp_with_highs

class HiGHSTreeEnv:
    """
    RL Environment wrapping the native C++ HiGHS solver.
    State: GraphInstance and its topological features.
    Action: Multipliers for the surrogate aggregation (K5 cliques).
    Reward: Reduction in simplex pivots and gap closure at the root relaxation.
    """
    def __init__(self, solver: TurboSolver, graphs: List[GraphInstance], max_cuts: int = 50):
        self.solver = solver
        self.graphs = graphs
        self.max_cuts = max_cuts
        self.current_idx = 0
        self.baselines = {}
        
    def reset(self) -> Tuple[GraphInstance, Dict[str, Any]]:
        self.current_idx = np.random.randint(len(self.graphs))
        self.graph = self.graphs[self.current_idx]
        
        if self.graph.name not in self.baselines:
            self._compute_baseline()
            
        return self.graph, {"baseline_pivots": self.baselines[self.graph.name]["pivots"]}

    def _compute_baseline(self):
        m = self.graph.num_edges
        c_weights = [-self.graph.weights.get(e, 1.0) for e in self.graph.edges]
        
        col_lower = [0.0] * m
        col_upper = [1.0] * m
        row_lower = []
        row_upper = []
        row_starts = [0]
        row_indices = []
        row_values = []
        integrality = [0] * m
        
        res_dict = solve_milp_with_highs(
            c_weights, col_lower, col_upper, row_lower, row_upper, 
            row_starts, row_indices, row_values, integrality
        )
        
        baseline_gap = -float(res_dict.get("fun", 0.0)) if res_dict.get("status") in (7, 9) else 0.0
        baseline_pivots = res_dict.get("simplex_iterations", 0)
        
        self.baselines[self.graph.name] = {
            "gap": baseline_gap,
            "pivots": baseline_pivots
        }
        
    def step(self, action_multipliers: Dict[Tuple[int, int, int, int, int], float]) -> Tuple[GraphInstance, float, bool, Dict[str, Any]]:
        m = self.graph.num_edges
        c_weights = [-self.graph.weights.get(e, 1.0) for e in self.graph.edges]
        col_lower = [0.0] * m
        col_upper = [1.0] * m
        integrality = [0] * m

        a_surr = np.zeros(m, dtype=np.float64)
        b_surr = 0.0
        
        edge_to_idx = {e: i for i, e in enumerate(self.graph.edges)}
        
        for clq, mult in action_multipliers.items():
            if mult > 1e-4:
                b_surr += 6.0 * mult
                import itertools
                for u, v in itertools.combinations(clq, 2):
                    e = (min(u, v), max(u, v))
                    if e in edge_to_idx:
                        a_surr[edge_to_idx[e]] += mult
                        
        row_lower = []
        row_upper = []
        row_starts = [0]
        row_indices = []
        row_values = []

        if b_surr > 1e-9:
            # Use exact rational verifier to verify the generated surrogate cut
            if self.solver.verifier is not None:
                mult_dict = {}
                for clq, mult in action_multipliers.items():
                    if mult > 1e-4:
                        mult_dict[tuple(clq)] = mult
                if mult_dict:
                    cert = self.solver.verifier.verify_clique_conic_combination(
                        self.graph, mult_dict
                    )
                    if not cert.is_valid:
                        return self.graph, -100.0, True, {
                            "error": f"Invalid surrogate cut generated: {cert.rejection_reason}",
                            "baseline_pivots": self.baselines[self.graph.name]["pivots"],
                            "accel_pivots": self.baselines[self.graph.name]["pivots"],
                            "baseline_gap": self.baselines[self.graph.name]["gap"],
                            "accel_gap": self.baselines[self.graph.name]["gap"],
                            "pivot_reduction": 0,
                            "gap_closure": 0
                        }
                    # Update b_surr from exact rational
                    b_surr = float(cert.exact_rhs)
            
            # Setup row
            nz_indices = np.where(a_surr > 1e-9)[0]
            if len(nz_indices) > 0:
                row_lower.append(-float('inf'))
                row_upper.append(float(b_surr))
                for idx in nz_indices:
                    row_indices.append(int(idx))
                    row_values.append(float(a_surr[idx]))
                row_starts.append(len(row_indices))
                        
        res_dict = solve_milp_with_highs(
            c_weights, col_lower, col_upper, row_lower, row_upper, 
            row_starts, row_indices, row_values, integrality
        )
            
        accel_gap = -float(res_dict.get("fun", 0.0)) if res_dict.get("status") in (7, 9) else 0.0
        accel_pivots = res_dict.get("simplex_iterations", 0)
        
        baseline = self.baselines[self.graph.name]
        
        pivot_reduction = max(0, baseline["pivots"] - accel_pivots)
        gap_closure = max(0, baseline["gap"] - accel_gap)
        
        reward = pivot_reduction + 10.0 * gap_closure
        
        info = {
            "baseline_pivots": baseline["pivots"],
            "accel_pivots": accel_pivots,
            "baseline_gap": baseline["gap"],
            "accel_gap": accel_gap,
            "pivot_reduction": pivot_reduction,
            "gap_closure": gap_closure
        }
        
        return self.graph, float(reward), True, info
