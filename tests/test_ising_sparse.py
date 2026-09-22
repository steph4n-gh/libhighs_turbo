"""Independent arithmetic, serialized proof validation, and large API routing."""

import copy
import time
from dataclasses import replace
from fractions import Fraction
from itertools import product

import numpy as np
import pytest
import scipy.sparse as sp

from highs_turbo import solve_ising, verify_ising_certificate
from highs_turbo.ising_cuts import _bound, make_certificate
from highs_turbo.ising_sparse import gram_lower_bound


def pack(matrix):
    matrix = sp.csr_matrix(matrix)
    return tuple(
        tuple(map(int, row)) for row in (matrix.indptr, matrix.indices, matrix.data)
    )


def test_sparse_gram_matches_python_integer_expansion_and_all_states():
    # Arbitrary, nontriangular, with an empty row and nonedge fill-in. Large
    # entries fit the row-norm proof but exceed the old n*max_entry**2 limit.
    factor = [
        [2**26, 0, 0, 0, 0],
        [2**25, -9, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [7, 4, 0, -5, 2],
        [-3, 0, 0, 0, 5],
    ]
    edges, weights, scale, den = (
        [(0, 1), (0, 4), (1, 3), (2, 4)],
        [3, -7, 2, 5],
        8,
        2**26,
    )
    bound = gram_lower_bound(edges, weights, scale, pack(factor), den)
    gram = [
        [sum(a * b for a, b in zip(left, right)) for right in factor] for left in factor
    ]
    expected = -Fraction(sum(abs(x) for row in gram for x in row), den**2)
    for (u, v), weight in zip(edges, weights):
        value = Fraction(2 * gram[u][v], den**2)
        expected += abs(value) - abs(Fraction(weight, scale) - value)
    assert bound == expected
    assert all(
        bound
        <= sum(Fraction(w, scale) * s[u] * s[v] for (u, v), w in zip(edges, weights))
        for s in product([-1, 1], repeat=5)
    )


def test_sparse_certificate_roundtrip_and_malformed_witnesses_without_optimizer(
    monkeypatch,
):
    import highs_turbo.ising_sparse as sparse
    import highs_turbo.ising_sdp as sdp
    import highspy

    edges = [(0, 1), (0, 2), (1, 2)]
    weights = dict.fromkeys(edges, Fraction(1))
    factor = pack([[4, 0, 0], [4, 1, 0], [4, 0, 1]])
    source = make_certificate([], [], edges, weights, Fraction())
    lower = _bound(
        (), (), source.denominator, edges, weights, Fraction(), (), 8, factor
    )
    proof = replace(
        source, lower_bound=lower, sparse_gram_factor=factor, gram_denominator=8
    )
    witness = proof.to_dict()
    assert witness["version"] == 4
    assert lower > -3
    for owner, name in [(sparse, "splu"), (sdp, "minimize"), (highspy.Highs, "run")]:
        monkeypatch.setattr(
            owner, name, lambda *args, **kwargs: pytest.fail("Checker called optimizer")
        )
    assert verify_ising_certificate({}, weights, witness)
    assert not verify_ising_certificate(
        {}, {**weights, (0, 1): Fraction(9, 8)}, witness
    )
    malformed = [
        [[0, 1, 3, 5], [0, 0, 0, 0, 2], [4, 4, 1, 4, 1]],  # duplicate column
        [[0, 1, 3, 5], [0, 0, 1, 0, 3], [4, 4, 1, 4, 1]],  # out of range
        [[0, 1, 3, 6], [0, 0, 1, 0, 2], [4, 4, 1, 4, 1]],  # wrong endpoint
        [[0, -1, 3, 5], [0, 0, 1, 0, 2], [4, 4, 1, 4, 1]],
        [[0, 1, 3, 5], [0, 0, 1, 0, 2], [4.0, 4, 1, 4, 1]],
        [[0, 1, 3, 5], [0, 0, 1, 0, 2], [True, 4, 1, 4, 1]],
        [[0, 1, 3, 5], [0, 0, 1, 0, 2], [4, 2**26, 2**26, 4, 1]],
        [[0, 1, 3, 5], [0, 0, 1, 0, 2], [2**100, 4, 1, 4, 1]],
    ]
    for bad in malformed:
        altered = copy.deepcopy(witness)
        altered["sparse_gram_factor"] = bad
        assert not verify_ising_certificate({}, weights, altered)
    for key, value in [
        ("gram_denominator", 0),
        ("lower_bound", [100, 1]),
        ("gram_factor", [[1], [0, 1], [0, 0, 1]]),
    ]:
        altered = {**witness, key: value}
        assert not verify_ising_certificate({}, weights, altered)


def test_sparse_expansion_budgets_reject_before_numeric_product(monkeypatch):
    import highs_turbo.ising_sparse as sparse

    edges = [(0, 1), (1, 2)]
    factor = pack(np.ones((3, 3), dtype=int))
    for name in ["MAX_FACTOR_ENTRIES", "MAX_GRAM_WORK", "MAX_GRAM_ENTRIES"]:
        with monkeypatch.context() as context:
            context.setattr(sparse, name, 1)
            with pytest.raises(ValueError):
                gram_lower_bound(edges, [1, 1], 1, factor, 16)


def test_reused_gram_still_validates_fields_and_complete_arithmetic(monkeypatch):
    import highs_turbo.ising_sparse as sparse
    from highs_turbo.ising_cuts import check_certificate

    edges = [(0, 1), (0, 2), (1, 2)]
    weights = dict.fromkeys(edges, Fraction(1))
    source = make_certificate([], [], edges, weights, Fraction())
    packed = pack([[4, 0, 0], [4, 1, 0], [4, 0, 1]])
    original = sparse._exact_gram_lower_bound
    computations = []

    def measured(*args):
        computations.append(args)
        return original(*args)

    monkeypatch.setattr(sparse, "_exact_gram_lower_bound", measured)
    with sparse._reuse_gram_bounds() as cache:
        lower = _bound((), (), source.denominator, edges, weights, Fraction(), (), 8, packed)
        proof = replace(source, lower_bound=lower, sparse_gram_factor=packed, gram_denominator=8)
        assert check_certificate(proof, edges, weights, Fraction())
        assert check_certificate(proof, edges, weights, Fraction())
        assert len(computations) == 1

        # These malformed values compare equal to their valid cache-key entries.
        for row, index, value in [(0, 0, False), (1, 0, 0.0), (2, 2, True)]:
            malformed = [list(values) for values in packed]
            malformed[row][index] = value
            altered = replace(proof, sparse_gram_factor=tuple(map(tuple, malformed)))
            assert not check_certificate(altered, edges, weights, Fraction())
        assert not check_certificate(replace(proof, gram_denominator=8.0), edges, weights, Fraction())
        assert len(computations) == 1

        changed = {**weights, (0, 1): Fraction(3, 2)}
        _bound((), (), source.denominator, edges, changed, Fraction(), (), 8, packed)
        other_factor = pack([[5, 0, 0], [4, 1, 0], [4, 0, 1]])
        _bound((), (), source.denominator, edges, weights, Fraction(), (), 8, other_factor)
        assert len(computations) == 3

        assert check_certificate(proof, edges, weights, Fraction())
        for name in ["MAX_FACTOR_ENTRIES", "MAX_GRAM_WORK", "MAX_GRAM_ENTRIES"]:
            with monkeypatch.context() as context:
                context.setattr(sparse, name, 1)
                assert not check_certificate(proof, edges, weights, Fraction())

    assert not cache
    assert sparse._gram_cache.get() is None
    before = len(computations)
    assert verify_ising_certificate({}, weights, proof.to_dict())
    assert len(computations) == before + 1


def test_gram_reuse_scopes_restore_and_clear_after_exception():
    import highs_turbo.ising_sparse as sparse

    arguments = ([(0, 1)], [1], 1, pack([[1, 0], [1, 1]]), 2)
    with sparse._reuse_gram_bounds() as outer:
        expected = gram_lower_bound(*arguments)
        with pytest.raises(RuntimeError, match="scope cleanup"):
            with sparse._reuse_gram_bounds() as inner:
                assert gram_lower_bound(*arguments) == expected
                assert len(inner) == 1
                raise RuntimeError("scope cleanup")
        assert not inner
        assert sparse._gram_cache.get() is outer
        assert gram_lower_bound(*arguments) == expected
        assert len(outer) == 1
    assert not outer
    assert sparse._gram_cache.get() is None


def test_default_large_graph_uses_sparse_certificate():
    # Independent triangles have a known exact minimum, without exponential
    # enumeration or reliance on a second solver for this large routing test.
    J = {
        (u + a, u + b): 1
        for u in range(0, 2052, 3)
        for a, b in [(0, 1), (0, 2), (1, 2)]
    }
    # Test proof routing without depending on initializer startup speed. The
    # checked-gap target stops before an unnecessary native integer solve.
    result = solve_ising({}, J, time_limit=10, certified_gap=500)
    assert result.certificate.sparse_gram_factor
    assert result.exact_cut_lower_bound <= -684 <= result.exact_energy
    assert result.exact_cut_lower_bound > -2052
    assert verify_ising_certificate({}, J, result.certificate.to_dict())


def test_colored_updates_preserve_unit_vectors_and_decrease_objective():
    from highs_turbo.ising_sparse import independent_groups, fit_vectors

    # Include an isolated vertex and a potential cut edge whose weight is
    # initially zero. Reusing these groups must remain valid when it activates.
    edges = [(0, 1), (1, 2), (0, 2), (2, 3)]
    groups = independent_groups(edges, 5)
    assert all(not ({u, v} <= set(group)) for u, v in edges for group in groups)
    vectors = np.random.default_rng(17).normal(size=(5, 3))
    vectors /= np.linalg.norm(vectors, axis=1)[:, None]
    isolated = vectors[4].copy()
    u, v = np.asarray(edges).T
    for weight in [0., -2.]:
        values = np.array([1., -1., weight, .5])
        matrix = sp.csr_matrix((np.r_[values, values],
                                (np.r_[u, v], np.r_[v, u])), shape=(5, 5))
        before = np.sum(vectors*(matrix @ vectors))
        vectors = fit_vectors(matrix, vectors, groups, deadline=time.perf_counter()+2)
        assert np.sum(vectors*(matrix @ vectors)) <= before+1e-12
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.)
        np.testing.assert_array_equal(vectors[4], isolated)
