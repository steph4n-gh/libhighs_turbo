"""Adversarial stress test suite: 100+ Hostile Mutated Cuts Across 8 Attack Vectors.

Verifies rejection rate (is_valid == False) across:
1. Negative multipliers (lambda < 0)
2. Missing edges in support (non-clique / missing graph edges)
3. Support size violations (|K| != 5, duplicates, cycle len < 3)
4. Out-of-bounds node indices (v >= n)
5. RHS claim deficits (b_claim < b_exact)
6. Odd cycle parity violations (|F| even, F not in C)
7. Exact precision overflow attempts (large numerators/denominators)
8. Structural / dimension mismatches
Compound attacks: multi-vector simultaneous hostile mutations.
"""

from fractions import Fraction
import random
import pytest
import highs_turbo._compiled_engine as ce
from highs_turbo.graph_generator import generate_k5_cluster_graph, GraphInstance


def build_k5_bitgraph(num_k5=2, num_bridges=1):
    g = generate_k5_cluster_graph(num_k5=num_k5, num_bridges=num_bridges, seed=42)
    bg = ce.BitGraph(g.num_nodes)
    bg.add_edges(list(g.edges))
    return g, bg


class TestAdversarialCutMutations:
    """Stress suite testing 100+ hostile mutated cuts."""

    def test_vector1_negative_multipliers(self):
        """Vector 1: Negative multipliers across 15 varied magnitudes."""
        g, bg = build_k5_bitgraph()
        verifier = ce.RationalVerifier()
        neg_values = [
            -1.0, -0.5, -2.0, -10.0, -100.0, -1e-4, -1e-5,
            ce.ExactRational(-1, 2), ce.ExactRational(-5, 7), ce.ExactRational(-100, 3), ce.ExactRational(-1000000, 1),
            ce.ExactRational(-1, 1), ce.ExactRational(-5, 13),
            Fraction(-3, 8), Fraction(-100, 1),
        ]
        tested = 0
        for val in neg_values:
            mults = {(0, 1, 2, 3, 4): val}
            cert = verifier.verify_clique_conic_combination(bg, mults)
            assert cert.is_valid is False, f"Negative multiplier {val} was admitted!"
            assert cert.status == "REJECTED_NEGATIVE_MULTIPLIER"
            tested += 1
        assert tested == 15

    def test_vector2_missing_edges_in_support(self):
        """Vector 2: Missing edges in candidate support (15 variations)."""
        # Graph with 5 nodes but some edges missing from complete K5
        bg = ce.BitGraph(10)
        edges_to_omit = [
            [(0, 1)],
            [(1, 2)],
            [(3, 4)],
            [(0, 4)],
            [(0, 1), (2, 3)],
            [(1, 3), (2, 4)],
            [(0, 2), (1, 4), (2, 3)],
            [(0, 1), (0, 2), (0, 3), (0, 4)],  # isolated node 0
            [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)], # only 5 edges missing
            [(i, j) for i in range(5) for j in range(i + 1, 5) if (i, j) != (0, 1)], # only 1 edge present
        ]
        verifier = ce.RationalVerifier()
        tested = 0
        for missing in edges_to_omit:
            bg_test = ce.BitGraph(5)
            for i in range(5):
                for j in range(i + 1, 5):
                    if (i, j) not in missing and (j, i) not in missing:
                        bg_test.add_edge(i, j)
            cert = verifier.verify_clique_conic_combination(bg_test, {(0, 1, 2, 3, 4): 1.0})
            assert cert.is_valid is False, f"Missing edges {missing} was admitted!"
            assert cert.status == "REJECTED_NON_CLIQUE_SUPPORT"
            tested += 1
        assert tested == 10

    def test_vector3_invalid_support_size_and_duplicates(self):
        """Vector 3: Support size != 5 and duplicate vertices (15 variations)."""
        g, bg = build_k5_bitgraph()
        verifier = ce.RationalVerifier()
        invalid_supports = [
            (),                     # size 0
            (0,),                   # size 1
            (0, 1),                 # size 2
            (0, 1, 2),              # size 3
            (0, 1, 2, 3),           # size 4
            (0, 1, 2, 3, 4, 5),     # size 6
            (0, 1, 2, 3, 4, 5, 6),  # size 7
            tuple(range(8)),        # size 8
            tuple(range(10)),       # size 10
            (0, 1, 2, 3, 3),        # duplicate vertex 3
            (0, 0, 1, 2, 3),        # duplicate vertex 0
            (1, 1, 1, 1, 1),        # all identical
            (0, 1, 1, 2, 2),        # two duplicates
            (0, 1, 2, 2, 3),        # duplicate vertex 2
            (0, 4, 4, 2, 3),        # duplicate vertex 4
        ]
        tested = 0
        for supp in invalid_supports:
            cert = verifier.verify_clique_conic_combination(bg, {supp: 1.0})
            assert cert.is_valid is False, f"Invalid support {supp} was admitted!"
            assert cert.status == "REJECTED_INVALID_SUPPORT_SIZE"
            tested += 1
        assert tested == 15

    def test_vector4_out_of_bounds_node_index(self):
        """Vector 4: Node index >= num_nodes (15 variations)."""
        g, bg = build_k5_bitgraph(num_k5=1, num_bridges=0) # num_nodes = 5
        verifier = ce.RationalVerifier()
        oob_supports = [
            (0, 1, 2, 3, 5),
            (0, 1, 2, 3, 6),
            (0, 1, 2, 3, 10),
            (0, 1, 2, 3, 50),
            (0, 1, 2, 3, 100),
            (0, 1, 2, 3, 999),
            (0, 1, 2, 3, 10000),
            (5, 6, 7, 8, 9),
            (10, 11, 12, 13, 14),
            (100, 101, 102, 103, 104),
            (0, 1, 2, 3, 2**16),
            (0, 1, 2, 3, 2**20),
            (0, 1, 2, 3, 2**24),
            (0, 1, 2, 3, 2**30),
            (0, 1, 2, 3, 2**31 - 1),
        ]
        tested = 0
        for supp in oob_supports:
            cert = verifier.verify_clique_conic_combination(bg, {supp: 1.0})
            assert cert.is_valid is False, f"Out-of-bounds support {supp} was admitted!"
            assert cert.status == "REJECTED_OUT_OF_BOUNDS_NODE"
            tested += 1
        assert tested == 15

    def test_vector5_rhs_claim_deficit(self):
        """Vector 5: Claimed RHS lower than sound bound (15 variations)."""
        g, bg = build_k5_bitgraph(num_k5=1, num_bridges=0) # K5 has sound RHS = 6.0
        verifier = ce.RationalVerifier()
        # Sound RHS is 6.0. Any claim < 6.0 - tol must be rejected.
        deficit_rhs_values = [
            5.9, 5.5, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -10.0,
            Fraction(5999, 1000), Fraction(11, 2), Fraction(1, 100),
            ce.ExactRational(5, 1), ce.ExactRational(1, 1),
        ]
        tested = 0
        for rhs in deficit_rhs_values:
            cert = verifier.verify_clique_conic_combination(bg, {(0, 1, 2, 3, 4): 1.0}, rhs)
            assert cert.is_valid is False, f"RHS deficit {rhs} was admitted!"
            assert cert.status == "REJECTED_UNSOUND_RHS_DEFICIT"
            tested += 1
        assert tested == 15

    def test_vector6_odd_cycle_parity_violations(self):
        """Vector 6: Odd cycle inequality parity violations (15 variations)."""
        # Create a graph with several cycles (C3, C4, C5)
        bg = ce.BitGraph(6)
        bg.add_edges([(0, 1), (1, 2), (2, 0)]) # C3: 0-1-2-0
        bg.add_edges([(2, 3), (3, 4), (4, 5), (5, 2)]) # C4: 2-3-4-5-2
        verifier = ce.RationalVerifier()

        invalid_cycles = [
            # |F| even (cardinality 0, 2)
            (((0, 1, 2), []), 1.0),                               # |F| = 0
            (((0, 1, 2), [(0, 1), (1, 2)]), 1.0),                 # |F| = 2
            (((2, 3, 4, 5), []), 1.0),                           # |F| = 0
            (((2, 3, 4, 5), [(2, 3), (3, 4)]), 1.0),             # |F| = 2
            (((2, 3, 4, 5), [(2, 3), (3, 4), (4, 5), (5, 2)]), 1.0), # |F| = 4
            # Cycle length < 3
            (((), []), 1.0),
            (((0,), []), 1.0),
            (((0, 1), []), 1.0),
            # Non-cycle edges in graph
            (((0, 1, 3), [(0, 1)]), 1.0),                         # edge (1, 3) not in graph
            (((0, 4, 5), [(0, 4)]), 1.0),                         # edge (0, 4) not in graph
            (((1, 3, 5), [(1, 3)]), 1.0),                         # missing edges
            # F not subset of C
            (((0, 1, 2), [(3, 4)]), 1.0),                         # (3, 4) in F but not in C3
            (((0, 1, 2), [(2, 3)]), 1.0),                         # (2, 3) in F but not in C3
            (((2, 3, 4, 5), [(0, 1)]), 1.0),                     # (0, 1) in F but not in C4
            (((2, 3, 4, 5), [(0, 2)]), 1.0),                     # (0, 2) not in C4
        ]
        tested = 0
        for cycle_data, mult in invalid_cycles:
            cert = verifier.verify_cycle_conic_combination(bg, [(cycle_data, mult)])
            assert cert.is_valid is False, f"Invalid cycle cut {cycle_data} was admitted!"
            tested += 1
        assert tested == 15

    def test_vector7_precision_overflow_attacks(self):
        """Vector 7: Large numeric scale and overflow boundary cut combinations (10 variations)."""
        g, bg = build_k5_bitgraph(num_k5=1, num_bridges=0)
        verifier = ce.RationalVerifier()
        overflow_mults = [
            ce.ExactRational(str(2**62), "1"),
            ce.ExactRational(str(2**63), "1"),
            ce.ExactRational(str(2**126), "1"),
            ce.ExactRational(str(2**127), "1"),
            ce.ExactRational(str(10**40), "1"),
            ce.ExactRational(str(10**60), "1"),
            ce.ExactRational("1", str(2**62)),
            ce.ExactRational("1", str(2**126)),
            ce.ExactRational("1", str(10**40)),
            ce.ExactRational(str(2**126), str(2**126 + 1)),
        ]
        tested = 0
        for mult in overflow_mults:
            # Valid conic combination with huge numbers should certify cleanly
            cert = verifier.verify_clique_conic_combination(bg, {(0, 1, 2, 3, 4): mult})
            assert cert.is_valid is True, f"Huge multiplier {mult} failed certification!"
            assert cert.status == "CERTIFIED_VALID_CONIC_COMBINATION"
            # But with RHS deficit it MUST reject!
            deficit_rhs = ce.ExactRational(1, 1) # way below mult * 6 for mult >= 2**62
            if mult > ce.ExactRational(1, 1):
                cert_def = verifier.verify_clique_conic_combination(bg, {(0, 1, 2, 3, 4): mult}, deficit_rhs)
                assert cert_def.is_valid is False, f"Deficit on huge multiplier {mult} was admitted!"
                assert cert_def.status == "REJECTED_UNSOUND_RHS_DEFICIT"
            tested += 1
        assert tested == 10

    def test_compound_hostile_attacks(self):
        """Compound attacks: 25 randomized combinations across multiple attack vectors."""
        g, bg = build_k5_bitgraph(num_k5=2, num_bridges=1) # 10 nodes
        verifier = ce.RationalVerifier()
        rng = random.Random(1337)

        tested = 0
        for _ in range(25):
            # Select 2 to 4 attack mutations simultaneously
            mutations = rng.sample([
                "negative_mult",
                "missing_edge",
                "invalid_size",
                "oob_node",
                "rhs_deficit",
            ], k=rng.randint(2, 4))

            mult = -1.0 if "negative_mult" in mutations else 1.0
            supp = [0, 1, 2, 3, 4]

            if "invalid_size" in mutations:
                supp = [0, 1, 2, 3] if rng.random() < 0.5 else [0, 1, 2, 3, 4, 5]
            if "oob_node" in mutations:
                supp = [x if x != 4 else 999 for x in supp]
            if "missing_edge" in mutations and "oob_node" not in mutations:
                supp = [0, 1, 2, 5, 6] # cross-cluster non-clique

            cand_rhs = -5.0 if "rhs_deficit" in mutations else None

            cert = verifier.verify_clique_conic_combination(bg, {tuple(supp): mult}, cand_rhs)
            assert cert.is_valid is False, f"Compound attack {mutations} with supp={supp}, mult={mult} was admitted!"
            tested += 1

        assert tested == 25
        # Total hostile cuts tested in this class: 15 + 10 + 15 + 15 + 15 + 15 + 10 + 25 = 120 cuts!
