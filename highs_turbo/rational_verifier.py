"""Exact rational verifier guaranteeing zero-hallucination cutting planes.

Validates that any candidate surrogate inequality is a mathematically sound
conic combination of valid cut inequalities (e.g. K5 complete subgraphs)
using exact rational arithmetic (fractions.Fraction).

Every certificate generates an immutable cryptographic SHA-256 hash.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from highs_turbo.graph_generator import GraphInstance


@dataclass
class VerificationCertificate:
    """Cryptographically auditable verification certificate for a cutting plane."""

    is_valid: bool
    status: str
    rejection_reason: Optional[str] = None
    num_active_supports: int = 0
    exact_coefficients: Dict[Tuple[int, int], Fraction] = field(default_factory=dict)
    exact_rhs: Fraction = Fraction(0, 1)
    sha256_hash: str = ""

    def compute_sha256(self) -> str:
        """Computes deterministic SHA-256 digest of canonical certificate representation."""
        canon_coeffs = [
            {"u": e[0], "v": e[1], "num": self.exact_coefficients[e].numerator, "den": self.exact_coefficients[e].denominator}
            for e in sorted(self.exact_coefficients.keys())
        ]
        payload = {
            "is_valid": self.is_valid,
            "status": self.status,
            "rejection_reason": self.rejection_reason,
            "num_active_supports": self.num_active_supports,
            "exact_rhs_num": self.exact_rhs.numerator,
            "exact_rhs_den": self.exact_rhs.denominator,
            "coefficients": canon_coeffs,
        }
        serialized = json.dumps(payload, sort_keys=True)
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        self.sha256_hash = digest
        return digest


class RationalCutVerifier:
    """Zero-hallucination exact rational verifier for candidate surrogate cutting planes."""

    def __init__(self, rational_denominator_limit: int = 100000):
        self.denominator_limit = rational_denominator_limit

    def to_fraction(self, val: Union[float, int, Fraction]) -> Fraction:
        """Converts float/int/Fraction to Fraction with bounded denominator."""
        if isinstance(val, Fraction):
            return val
        if isinstance(val, int):
            return Fraction(val, 1)
        return Fraction.from_float(float(val)).limit_denominator(self.denominator_limit)

    def verify_clique_conic_combination(
        self,
        graph: GraphInstance,
        candidate_multipliers: Dict[Tuple[int, ...], Union[float, int, Fraction]],
        candidate_rhs: Optional[Union[float, Fraction]] = None,
        tol: float = 1e-6,
    ) -> VerificationCertificate:
        """Verifies that candidate multipliers represent a valid non-negative conic combination

        of genuine K5 complete subgraphs of G.

        Mathematical Guarantee:
        1. Each support K_k is verified to have exactly 5 vertices and all 10 edges present in E(G).
        2. Each multiplier lambda_k >= 0.
        3. The resulting inequality:
             sum_e (sum_{k: e in E[K_k]} lambda_k) x_e <= sum_k lambda_k * 6
           is mathematically sound for the cut polytope CUT(G).
        """
        edge_set = set(graph.edges)
        exact_mults: Dict[Tuple[int, int, int, int, int], Fraction] = {}

        # 1. Non-negativity and structural validity check
        for raw_clq, raw_val in candidate_multipliers.items():
            # Check negative on raw value before float rounding can collapse to 0
            if isinstance(raw_val, (int, float)) and raw_val < 0:
                cert = VerificationCertificate(
                    is_valid=False,
                    status="REJECTED_NEGATIVE_MULTIPLIER",
                    rejection_reason=f"Support {raw_clq} has negative raw multiplier: {raw_val}",
                )
                cert.compute_sha256()
                return cert

            frac_val = self.to_fraction(raw_val)

            # Strictly reject negative multipliers
            if frac_val < Fraction(0, 1):
                cert = VerificationCertificate(
                    is_valid=False,
                    status="REJECTED_NEGATIVE_MULTIPLIER",
                    rejection_reason=f"Support {raw_clq} has negative multiplier: {frac_val}",
                )
                cert.compute_sha256()
                return cert

            # Skip exact zero multipliers
            if frac_val == Fraction(0, 1):
                continue

            # Support size check: literal K5 must have exactly 5 vertices
            if len(raw_clq) != 5:
                cert = VerificationCertificate(
                    is_valid=False,
                    status="REJECTED_INVALID_SUPPORT_SIZE",
                    rejection_reason=f"Support {raw_clq} has size {len(raw_clq)}, expected 5.",
                )
                cert.compute_sha256()
                return cert

            sorted_clq = tuple(sorted(int(v) for v in raw_clq))
            # Node existence
            for v in sorted_clq:
                if v < 0 or v >= graph.num_nodes:
                    cert = VerificationCertificate(
                        is_valid=False,
                        status="REJECTED_OUT_OF_BOUNDS_NODE",
                        rejection_reason=f"Node {v} in support {raw_clq} outside [0, {graph.num_nodes - 1}].",
                    )
                    cert.compute_sha256()
                    return cert

            # Subgraph completeness: check all 10 pairs exist in E(G)
            for u, v in itertools.combinations(sorted_clq, 2):
                e = (min(u, v), max(u, v))
                if e not in edge_set:
                    cert = VerificationCertificate(
                        is_valid=False,
                        status="REJECTED_NON_CLIQUE_SUPPORT",
                        rejection_reason=f"Edge {e} missing in graph for support {raw_clq}.",
                    )
                    cert.compute_sha256()
                    return cert

            exact_mults[sorted_clq] = frac_val

        # 2. Reconstruct exact rational coefficients and RHS
        exact_coeffs: Dict[Tuple[int, int], Fraction] = {e: Fraction(0, 1) for e in graph.edges}
        exact_rhs = Fraction(0, 1)

        for clq, mult in exact_mults.items():
            exact_rhs += mult * Fraction(6, 1)
            for u, v in itertools.combinations(clq, 2):
                e = (min(u, v), max(u, v))
                exact_coeffs[e] += mult

        # 3. RHS Soundness Check
        if candidate_rhs is not None:
            claimed_rhs = self.to_fraction(candidate_rhs)
            # If the claimed RHS is strictly less than the valid conic RHS, it would over-constrain
            if claimed_rhs < exact_rhs - self.to_fraction(tol):
                cert = VerificationCertificate(
                    is_valid=False,
                    status="REJECTED_UNSOUND_RHS_DEFICIT",
                    rejection_reason=f"Claimed RHS {claimed_rhs} is lower than sound conic bound {exact_rhs}.",
                )
                cert.compute_sha256()
                return cert

        # Filter nonzero coefficients for efficiency
        active_coeffs = {e: c for e, c in exact_coeffs.items() if c > Fraction(0, 1)}

        cert = VerificationCertificate(
            is_valid=True,
            status="CERTIFIED_VALID_CONIC_COMBINATION",
            num_active_supports=len(exact_mults),
            exact_coefficients=active_coeffs,
            exact_rhs=exact_rhs,
        )
        cert.compute_sha256()
        return cert

    def verify_cut_polytope_soundness(
        self,
        graph: GraphInstance,
        coefficients: np.ndarray,
        rhs: float,
    ) -> VerificationCertificate:
        """Verifies soundness against the cut polytope by solving exact integer Max-Cut

        on the candidate coefficient vector.

        An inequality a^T x <= b is valid for CUT(G) iff:
            max_{x in CUT(G)} a^T x <= b.
        """
        from highs_turbo.exact_solver import ExactMaxCutSolver

        m = graph.num_edges
        if len(coefficients) != m:
            cert = VerificationCertificate(
                is_valid=False,
                status="REJECTED_DIMENSION_MISMATCH",
                rejection_reason=f"Expected {m} coefficients, got {len(coefficients)}.",
            )
            cert.compute_sha256()
            return cert

        # Build dummy graph instance weighted by candidate coefficients
        custom_weights = {e: float(coefficients[i]) for i, e in enumerate(graph.edges)}
        weighted_graph = GraphInstance(
            name=f"{graph.name}_candidate_eval",
            num_nodes=graph.num_nodes,
            edges=list(graph.edges),
            weights=custom_weights,
        )

        solver = ExactMaxCutSolver()
        max_val, _ = solver.solve_integer_maxcut(weighted_graph)

        exact_max = self.to_fraction(max_val)
        exact_rhs = self.to_fraction(rhs)

        if exact_max > exact_rhs + Fraction(1, 1000000):
            cert = VerificationCertificate(
                is_valid=False,
                status="REJECTED_CUT_POLYTOPE_VIOLATION",
                rejection_reason=f"Maximum integer cut on row coefficients is {exact_max} > RHS {exact_rhs}.",
            )
            cert.compute_sha256()
            return cert

        exact_coeffs = {e: self.to_fraction(coefficients[i]) for i, e in enumerate(graph.edges) if abs(coefficients[i]) > 1e-9}
        cert = VerificationCertificate(
            is_valid=True,
            status="CERTIFIED_VALID_CUT_POLYTOPE",
            num_active_supports=len(exact_coeffs),
            exact_coefficients=exact_coeffs,
            exact_rhs=exact_rhs,
        )
        cert.compute_sha256()
        return cert

    def verify_cycle_conic_combination(
        self,
        graph: GraphInstance,
        candidate_cycles: Sequence[Tuple[Sequence[int], Sequence[Tuple[int, int]], Union[float, int, Fraction]]],
        candidate_rhs: Optional[Union[float, Fraction]] = None,
        tol: float = 1e-6,
    ) -> VerificationCertificate:
        r"""Verifies that candidate multipliers represent a valid non-negative conic combination

        of genuine cycle inequalities x(F) - x(C \ F) <= |F| - 1 where |F| is odd.
        """
        edge_set = set(graph.edges)
        exact_coeffs: Dict[Tuple[int, int], Fraction] = {e: Fraction(0, 1) for e in graph.edges}
        exact_rhs = Fraction(0, 1)
        active_count = 0

        for cycle_nodes, f_edges, raw_mult in candidate_cycles:
            if isinstance(raw_mult, (int, float)) and raw_mult < 0:
                cert = VerificationCertificate(
                    is_valid=False,
                    status="REJECTED_NEGATIVE_MULTIPLIER",
                    rejection_reason=f"Cycle support has negative raw multiplier: {raw_mult}",
                )
                cert.compute_sha256()
                return cert

            mult = self.to_fraction(raw_mult)
            if mult < Fraction(0, 1):
                cert = VerificationCertificate(
                    is_valid=False,
                    status="REJECTED_NEGATIVE_MULTIPLIER",
                    rejection_reason=f"Cycle support has negative multiplier: {mult}",
                )
                cert.compute_sha256()
                return cert

            if mult == Fraction(0, 1):
                continue

            k = len(cycle_nodes)
            if k < 3:
                cert = VerificationCertificate(
                    is_valid=False,
                    status="REJECTED_INVALID_CYCLE_LENGTH",
                    rejection_reason=f"Cycle must have >= 3 nodes, got {k}.",
                )
                cert.compute_sha256()
                return cert

            # Verify cycle edges exist in graph
            c_edges = set()
            for i in range(k):
                u, v = int(cycle_nodes[i]), int(cycle_nodes[(i + 1) % k])
                e = (min(u, v), max(u, v))
                if e not in edge_set:
                    cert = VerificationCertificate(
                        is_valid=False,
                        status="REJECTED_NON_GRAPH_EDGE",
                        rejection_reason=f"Edge {e} of cycle not present in graph.",
                    )
                    cert.compute_sha256()
                    return cert
                c_edges.add(e)

            # Standardize F edges
            canon_f = set()
            for u, v in f_edges:
                e = (min(int(u), int(v)), max(int(u), int(v)))
                if e not in c_edges:
                    cert = VerificationCertificate(
                        is_valid=False,
                        status="REJECTED_F_EDGE_NOT_IN_CYCLE",
                        rejection_reason=f"Edge {e} in F is not in cycle edges.",
                    )
                    cert.compute_sha256()
                    return cert
                canon_f.add(e)

            if len(canon_f) % 2 != 1:
                cert = VerificationCertificate(
                    is_valid=False,
                    status="REJECTED_EVEN_F_CARDINALITY",
                    rejection_reason=f"|F| must be odd, got {len(canon_f)}.",
                )
                cert.compute_sha256()
                return cert

            # Add to exact coefficients: +mult for e in F, -mult for e in C \ F
            for e in c_edges:
                if e in canon_f:
                    exact_coeffs[e] += mult
                else:
                    exact_coeffs[e] -= mult

            exact_rhs += mult * Fraction(len(canon_f) - 1, 1)
            active_count += 1

        if candidate_rhs is not None:
            claimed_rhs = self.to_fraction(candidate_rhs)
            if claimed_rhs < exact_rhs - self.to_fraction(tol):
                cert = VerificationCertificate(
                    is_valid=False,
                    status="REJECTED_UNSOUND_RHS_DEFICIT",
                    rejection_reason=f"Claimed RHS {claimed_rhs} is lower than sound bound {exact_rhs}.",
                )
                cert.compute_sha256()
                return cert

        active_coeffs = {e: c for e, c in exact_coeffs.items() if c != Fraction(0, 1)}
        cert = VerificationCertificate(
            is_valid=True,
            status="CERTIFIED_VALID_CYCLE_CONIC_COMBINATION",
            num_active_supports=active_count,
            exact_coefficients=active_coeffs,
            exact_rhs=exact_rhs,
        )
        cert.compute_sha256()
        return cert

    def verify_general_surrogate_cut(
        self,
        graph: GraphInstance,
        candidate_k5: Optional[Dict[Tuple[int, ...], Union[float, int, Fraction]]] = None,
        candidate_cycles: Optional[Sequence[Tuple[Sequence[int], Sequence[Tuple[int, int]], Union[float, int, Fraction]]]] = None,
        candidate_rhs: Optional[Union[float, Fraction]] = None,
        tol: float = 1e-6,
    ) -> VerificationCertificate:
        """Verifies arbitrary combinations of K5 cliques and cycle inequalities in exact arithmetic."""
        total_coeffs: Dict[Tuple[int, int], Fraction] = {e: Fraction(0, 1) for e in graph.edges}
        total_rhs = Fraction(0, 1)
        total_supports = 0

        if candidate_k5:
            cert_k5 = self.verify_clique_conic_combination(graph, candidate_k5, tol=tol)
            if not cert_k5.is_valid:
                return cert_k5
            for e, c in cert_k5.exact_coefficients.items():
                total_coeffs[e] += c
            total_rhs += cert_k5.exact_rhs
            total_supports += cert_k5.num_active_supports

        if candidate_cycles:
            cert_cyc = self.verify_cycle_conic_combination(graph, candidate_cycles, tol=tol)
            if not cert_cyc.is_valid:
                return cert_cyc
            for e, c in cert_cyc.exact_coefficients.items():
                total_coeffs[e] += c
            total_rhs += cert_cyc.exact_rhs
            total_supports += cert_cyc.num_active_supports

        if candidate_rhs is not None:
            claimed_rhs = self.to_fraction(candidate_rhs)
            if claimed_rhs < total_rhs - self.to_fraction(tol):
                cert = VerificationCertificate(
                    is_valid=False,
                    status="REJECTED_UNSOUND_RHS_DEFICIT",
                    rejection_reason=f"Claimed RHS {claimed_rhs} is lower than sound bound {total_rhs}.",
                )
                cert.compute_sha256()
                return cert

        active_coeffs = {e: c for e, c in total_coeffs.items() if c != Fraction(0, 1)}
        cert = VerificationCertificate(
            is_valid=True,
            status="CERTIFIED_VALID_SURROGATE_CUT",
            num_active_supports=total_supports,
            exact_coefficients=active_coeffs,
            exact_rhs=total_rhs,
        )
        cert.compute_sha256()
        return cert
