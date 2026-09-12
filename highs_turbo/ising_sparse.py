"""Sparse sum-of-squares proposals and exact integer Gram verification.

The numerical factorization proposes B/D. The checker only uses the fact
that s' B B' s / D**2 is nonnegative, and accounts for every residual entry.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from fractions import Fraction


MAX_SPARSE_VERTICES = 8192
MAX_FACTOR_ENTRIES = 8_000_000
MAX_GRAM_ENTRIES = 16_000_000
MAX_GRAM_WORK = 2_000_000_000


def factor_witness(matrix, vectors, magnitude=1.0):
    """Propose a sparse dyadic square without a dense eigenvalue solve."""
    n = matrix.shape[0]
    if not 1 < n <= MAX_SPARSE_VERTICES:
        raise ValueError("Unsupported sparse factor size")
    diagonal = np.sum(vectors * (matrix @ vectors), axis=1)
    slack = (matrix - sp.diags(diagonal)).tocsc()
    shift = max(1.0, float(np.max(abs(slack.diagonal())))) * 1e-6
    for _ in range(9):
        lu = splu(
            slack + sp.eye(n, format="csc") * shift,
            permc_spec="MMD_AT_PLUS_A",
            diag_pivot_thresh=0,
            options={"SymmetricMode": True, "Equil": False},
        )
        if lu.L.nnz > MAX_FACTOR_ENTRIES:
            raise ValueError("Sparse factor exceeds the storage budget")
        diagonal = lu.U.diagonal()
        if np.min(diagonal) > 0 and np.array_equal(lu.perm_r, lu.perm_c):
            break
        shift *= 4
    else:
        raise ValueError("Could not construct a stable sparse square")
    # SuperLU's Pr A Pc = L U convention maps original row i to perm_r[i].
    # Positive pivots and symmetry only guide proposal quality; no numerical
    # factorization identity or definiteness assertion is trusted below.
    factor = sp.csr_matrix(
        (lu.L @ sp.diags(np.sqrt(diagonal * magnitude))).tocsr()[lu.perm_r, :]
    )
    norm = float(np.max(np.asarray(factor.multiply(factor).sum(axis=1))))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Invalid numerical sparse factor")
    bits = min(24, int(np.floor(np.log2(np.sqrt(2**50 / norm)))))
    if bits < 0:
        raise ValueError("Sparse factor exceeds the exact integer range")
    denominator = 2**bits
    factor.data = np.rint(factor.data * denominator).astype(np.int64)
    factor.eliminate_zeros()
    factor.sort_indices()
    return (
        tuple(map(int, factor.indptr)),
        tuple(map(int, factor.indices)),
        tuple(map(int, factor.data)),
    ), denominator


def _unpack_factor(packed, n):
    """Validate all CSR structure and integer ranges before sparse arithmetic."""
    if (
        not 0 < n <= MAX_SPARSE_VERTICES
        or not isinstance(packed, tuple)
        or len(packed) != 3
    ):
        raise ValueError("Invalid sparse Gram dimensions")
    starts, indices, values = packed
    if (
        any(not isinstance(row, tuple) for row in packed)
        or len(starts) != n + 1
        or len(indices) != len(values)
        or len(values) > MAX_FACTOR_ENTRIES
        or any(type(x) is not int for row in packed for x in row)
        or starts[0] != 0
        or starts[-1] != len(values)
        or any(a > b or a < 0 for a, b in zip(starts, starts[1:]))
        or any(not 0 <= x < n for x in indices)
    ):
        raise ValueError("Invalid sparse Gram structure")
    maximum = max(map(abs, values), default=0)
    length = max((b - a for a, b in zip(starts, starts[1:])), default=0)
    if length * maximum**2 >= 2**63:
        raise ValueError("Sparse row norm accumulation exceeds int64")
    # Construct arrays only after proving their entries fit. Reject duplicates
    # rather than allowing a library conversion to add them before checking.
    columns = np.asarray(indices, dtype=np.int32)
    for begin, end in zip(starts, starts[1:]):
        if end - begin > 1 and np.any(
            columns[begin + 1 : end] <= columns[begin : end - 1]
        ):
            raise ValueError("Sparse Gram columns must be strictly increasing")
    return sp.csr_matrix(
        (
            np.asarray(values, dtype=np.int64),
            columns,
            np.asarray(starts, dtype=np.int32),
        ),
        shape=(n, n),
    )


def gram_lower_bound(edges, residual, scale, packed, denominator):
    """Exactly bound the residual quadratic form, including nonedge fill-in.

    Each integer row has squared norm < 2**53. Cauchy-Schwarz then proves
    sum_k |B_ik B_jk| < 2**53 for every row pair. Every product and every
    possible partial sum is therefore an exactly representable binary64
    integer. Standard CPU sparse multiplication computes the exact Gram.
    """
    n = max((max(edge) for edge in edges), default=-1) + 1
    if type(denominator) is not int or denominator <= 0:
        raise ValueError("Expected a positive integer Gram denominator")
    factor = _unpack_factor(packed, n)
    norms = np.asarray(factor.multiply(factor).sum(axis=1)).ravel()
    if np.max(norms, initial=0) >= 2**53:
        raise ValueError("Sparse Gram products exceed the exact binary64 range")
    # A factor column contributes an outer product. Check work before the
    # symbolic product, and expanded storage before numerical multiplication.
    degrees = np.bincount(factor.indices, minlength=n)
    if int(degrees @ degrees) > MAX_GRAM_WORK:
        raise ValueError("Sparse Gram expansion exceeds the work budget")
    pattern = factor.astype(bool)
    expanded = pattern @ pattern.T
    if expanded.nnz > MAX_GRAM_ENTRIES:
        raise ValueError("Sparse Gram pattern exceeds the storage budget")
    del pattern, expanded
    floats = factor.astype(np.float64)
    gram = (floats @ floats.T).tocsr()
    gram.data = gram.data.astype(np.int64)
    gram.sort_indices()
    maximum = int(np.max(np.abs(gram.data), initial=0))
    if n * maximum >= 2**63:
        raise ValueError("Sparse Gram row absolute sum exceeds int64")
    total = sum(map(int, np.asarray(abs(gram).sum(axis=1)).ravel()))
    square = denominator * denominator
    numerator = -total * scale
    u, v = np.asarray(edges, dtype=int).reshape(-1, 2).T
    values = np.asarray(gram[u, v]).ravel()
    for weight, value in zip(residual, values):
        value = int(value)
        numerator += 2 * abs(value) * scale - abs(weight * square - 2 * value * scale)
    return Fraction(numerator, scale * square)
