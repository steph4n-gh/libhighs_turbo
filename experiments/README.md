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

Python SOS research additionally requires `cvxpy` and its chosen solver. Julia
experiments require the research environment described in the results document.
These are optional research dependencies, not package runtime dependencies.

The tested higher-order lifts and larger nonlocal parity/region cuts did not
produce enough additional checked-bound improvement to justify production
integration. The useful geometric triangle separator was moved into
`highs_turbo/ising_geometry.py`; no duplicate implementation remains here.
