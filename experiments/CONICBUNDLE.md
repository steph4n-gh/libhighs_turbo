# ConicBundle fixed-cut comparison

This optional research driver links the unmodified ConicBundle 1.a.2 library.
It is not linked into, imported by, or required to install `highs_turbo`.
ConicBundle is GPL-3.0-or-later; its license applies to that external library
and binaries linked with it. No ConicBundle source or binary is vendored here.

Download the authors' archive from
[the official distribution](https://www-user.tu-chemnitz.de/~helmberg/ConicBundle/CB_v1.a.2.tgz).
The measured archive's SHA-256 is
`2baec791c00dbee164c5cd97b143206de37742c6263a1105282c311da0051eef`.
The following builds reproduce the optimized Apple Clang / Accelerate
configuration used for the reported measurements:

```bash
CB_ROOT=/path/to/extracted/ConicBundle
CB_BLAS_INCLUDE="$(xcrun --show-sdk-path)/System/Library/Frameworks/Accelerate.framework/Frameworks/vecLib.framework/Headers"
make -C "$CB_ROOT" -j4 MODE=OPTI CXX=clang++ CC=clang \
  CXXFLAGS="-O3 -DNDEBUG -std=c++11 -DWITH_BLAS -Wno-deprecated-declarations -I$CB_BLAS_INCLUDE" \
  DFLAGS=-MM AR=ar ARFLAGS=cr RANLIB=ranlib lib/libcb.a
clang++ -O3 -DNDEBUG -std=c++11 -DWITH_BLAS -Wno-deprecated-declarations \
  -I"$CB_ROOT/include" -I"$CB_ROOT/CBsources" -I"$CB_ROOT/Matrix" -I"$CB_ROOT/Tools" \
  -I"$CB_BLAS_INCLUDE" experiments/conicbundle_fixed.cxx \
  "$CB_ROOT/lib/libcb.a" -framework Accelerate -o /tmp/conicbundle-fixed
```

First generate the frozen sparse-geometric inputs and the authors' default
Mixing vectors using the commands in
[`SPARSE_GEOMETRIC_RESULTS.md`](../SPARSE_GEOMETRIC_RESULTS.md). Then run:

```bash
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m experiments.conicbundle_suite \
  --directory /tmp/sparse-geometry-final --binary /tmp/conicbundle-fixed \
  --seconds 10 --cases pegasus-7111 G56
```

Each case runs with zero initial multipliers and with diagonal multipliers
computed from the authors' default Mixing vectors. The latter's vector solve
is not charged. Reading those vectors and forming the multipliers are charged.
All selected cuts are also supplied free. No rounding heuristic or integer
solve is requested from this baseline. This favors its bound-construction
cost and isolates the constrained SDP optimizer from the cut separator.

The driver maximizes the negative Ising objective. For a cut
`sum_e a_e x_e <= b`, the correlation condition is
`sum_e a_e X_e >= sum_e a_e - 2*b`. ConicBundle's diagonal multipliers are
unrestricted, its cut multipliers are nonnegative, and its spectral objective
has trace factor `n`. The library receives a `PSCAffineFunction` and runs its
ordinary spectral bundle algorithm with relative tolerance `1e-7`.
Its default random generator seed is retained. Its ten-second limit uses
user CPU time; measured wall times and final certificate cost are reported.

The native program exports its dual multipliers and approximate primal
vectors. The Python runner reconstructs the spectral shift from the reported
objective and supplies both a dual-slack proposal and a normalized-vector
proposal to the same sparse exact checker used for AugmentedMixing. Neither
numerical eigenvalue claims nor the solver's reported bound are trusted.
Every repair, failed repair, and the best accepted bound are recorded. The
reported total charges both proposals and their independent checks.

This is a comparison with the ConicBundle library on fixed triangle sets,
not its tutorial's complete dynamic triangle-separation application or a
fully tuned branch-and-cut solver. The small sign-convention sanity check
uses a unit-weight triangle plus its triangle cut: the driver returns a
numerical minimum near -1, matching its exhaustively known optimum.
