# Research prototypes

These modules support the higher-order certificate investigation. Production
code does not import them. They are retained to make both successful and
unsuccessful directions inspectable; they are not additional public APIs.

- `lifted_ising.py`: Boolean monomials represented by bit masks, exact SOS
  coefficient expansion, CVXPY and direct ADMM formulations, selected products.
- `lifted_mixing.py`: low-rank constrained fitting for selected monomial bases.
- `parity_ising.py`: geometric proposals for larger odd-parity supports.
- `nonlocal_regions.py`: exact local spin-polytope cuts on geometric supports.
- `published_baselines.jl`: unmodified AugmentedMixing and TSSOS runners with a
  separate small JIT warmup. TSSOS uses its supported SCS backend.
- `check_published.py`: exact repair and objective-lattice rounding for published
  solver output, retaining the best of multiple repairs and the trivial bound.
- `geometric_suite.py`, `published_suite.py`: reproduce the frozen comparison
  documented in `GEOMETRIC_ISING_RESULTS.md`.
- `geometric_suite.py --suite sparse`, `original_mixing_suite.py`: larger
  before/after and unmodified 2017 Mixing comparisons, recorded in
  `SPARSE_ISING_RESULTS.md`.
- `colored_mixing.py`: research on vectorizing known coordinate minimizers
  over graph color classes, using the existing NumPy/SciPy dependencies.

Python SOS research additionally requires `cvxpy` and its chosen solver. Julia
experiments require the research environment described in the results document.
These are optional research dependencies, not package runtime dependencies.

The tested higher-order lifts and larger nonlocal parity/region cuts did not
produce enough additional checked-bound improvement to justify production
integration. The useful geometric triangle separator was moved into
`highs_turbo/ising_geometry.py`; no duplicate implementation remains here.
Coupled odd-triple monomials can close a perturbed five-spin example that
our tested pair-monomial run left open. Their tested larger-instance gains were
too small to promote. Sparse factor proposals and their integer-range checker
were moved into `highs_turbo/ising_sparse.py`.
