# nested_sampler

Nested sampling for feasibility analysis: a standalone port of the
`mc::NSFEAS` sampler of the MAGNUS library 
https://github.com/omega-icl/magnus (for licensing matters consider the license of the magnus repository!)
(feasibility phase only, no MC++
DAG machinery). Given a box domain and a function that returns constraint
values, it returns the live, dead and discarded points.

The sampler reproduces MAGNUS point for point: the same Sobol sequence for the
initial live points and box proposals, the same pseudo-random stream for
ellipsoid proposals, the same enclosing-ellipsoid construction, the same
acceptance and termination rules. See "Compatibility with MAGNUS" below.

## Installation

Requirements: Python 3.9+, numpy, a C++ compiler (GCC, Clang, or MSVC Build
Tools on Windows), CMake 3.15+ (fetched automatically by pip if missing).

```
pip install .
```

That builds the extension with scikit-build-core and installs the package.
For development, a plain CMake build drops the extension into `python/nested_sampler/`:

```
pip install pybind11
cmake -B build
cmake --build build --config Release
set PYTHONPATH=python      (Windows)   or   export PYTHONPATH=python
```

Tests: `pip install pytest` then `pytest tests`.

## Usage

```python
import numpy as np
from nested_sampler import NestedSampler

def constraints(d, p=None):
    p0 = 1.0 if p is None else p[0]
    y = p0 * d[0] ** 2 + d[1]
    return [y - 0.75, 0.20 - y]        # feasible iff every value <= 0

# Nominal parameter
result = NestedSampler(constraints, lb=[-1, -1], ub=[1, 1], num_live=500).sample()
print(result.summary())
result.live.x          # (500, 2) live points, all feasible when result.converged
result.live.value      # their criterion (max constraint value), ascending
result.dead.x          # dead points in order of death
result.discarded.x     # rejected proposals
result.history.contour # contour level per iteration
result.save("run.npz") # Result.load("run.npz") reads it back

# Uncertain parameter: scenarios with a value-at-risk criterion
scenarios = np.random.default_rng(0).normal(1.0, 0.5, size=(100, 1))
result = NestedSampler(constraints, lb=[-1, -1], ub=[1, 1], scenarios=scenarios,
                       criterion="var", threshold=0.05, num_live=500).sample()
result.live.prob       # probability of feasibility of each live point across the scenarios
```

The constraint function is called as `constraints(x)` without scenarios and
`constraints(x, p)` for every scenario row `p` otherwise. With
`vectorized=True` it receives all points at once, `constraints(X)` or
`constraints(X, P)`, and must return an array of shape `(m, ng)` or
`(m, ns, ng)`. An exception or a NaN marks a failed evaluation; the point is
dropped and counted in `result.stats.failures`.

Options (MAGNUS names in brackets): `num_live` [NUMLIVE], `num_prop`
[NUMPROP], `criterion` "var" or "cvar" [FEASCRIT], `threshold` [FEASTHRES],
`ell_mag` [ELLMAG], `ell_red` [ELLRED], `max_iter` [MAXITER], `max_err`
[MAXERR], `max_time` [MAXCPU], `display` [DISPLEVEL], `display_iter`
[DISPITER], `seed` (pseudo-random seed, default 5489 = Armadillo default),
`rng_skip` (advanced, see below).

The C++ core can be used on its own; see `examples/cpp_example.cpp` and
`include/nested_sampler/nested_sampler.hpp`.

## Algorithm

1. Draw `num_live` points from the box domain with a 64-bit Sobol sequence and
   evaluate their criterion: the maximal constraint value, or with scenarios
   its value-at-risk (VaR) or conditional value-at-risk (CVaR) at the given
   percentile. A point is feasible iff its criterion is <= 0.
2. Repeat until the worst live point is feasible:
   - build the nest from the live points: their bounding box (padded by the
     magnification factor and clipped to the domain) and the enclosing
     ellipsoid (mean, Cholesky factor of the covariance, scaled to the farthest
     live point and inflated by the factor);
   - draw `num_prop` proposals: from the box with the Sobol sequence when the
     box is not larger than the ellipsoid, otherwise uniformly from the
     ellipsoid with points outside the domain rejected;
   - evaluate all proposals, then process them in order: a proposal whose
     criterion does not exceed the worst live point replaces it (the worst
     point dies); the others are discarded. After each replacement the
     magnification factor becomes `ell_mag * exp(-ell_red * n_dead / num_live)`.

## Compatibility with MAGNUS

Verified point for point against MAGNUS on a set of test problems during
development (nominal and scenario-based, VaR and CVaR, 2-D and 3-D).

What is replicated:

- Boost's 64-bit Sobol engine (Joe-Kuo direction numbers, Gray-code order,
  zero point skipped), reimplemented in `sobol.hpp`. The direction-number
  table (`sobol_table.hpp`, up to 512 dimensions) is extracted from the Boost
  header by `tools/extract_sobol_table.py`, so Boost is not needed to build.
- Armadillo's pseudo-random stream on Linux: `std::mt19937_64` (identical on
  every platform) with the libstdc++ uniform and normal deviates written out
  in `rng.hpp`, so MSVC builds produce the same numbers.
- The operand order of the reference build for the ellipsoid draw (radius
  before direction), the fresh normal distribution per draw, the covariance
  normalised by N, the multimap tie-breaking, and the batch semantics: every
  proposal of a batch is processed even after the stop level is reached.
- MAGNUS reseeds the Sobol sequence at every call but never reseeds the
  Armadillo stream, so a second `sample()` in one MAGNUS process starts from
  a different pseudo-random state. Here every run starts from `seed`; use
  `rng_skip` to discard draws if you need to align with such a state.

Deliberate deviations:

- The wall-time limit works as intended. In MAGNUS the check in `_terminate`
  has an inverted condition and interrupts immediately when MAXCPU is set.
- Failed evaluations in the proposal loop stay paired with their points. In
  MAGNUS's single-threaded mode a failed proposal shifts the values of the
  following proposals onto the wrong points.
- `max_err` is also enforced during the main loop, not only during
  initialisation, and `evaluations` counts every attempted point.
- The likelihood (evidence) phase of NSFEAS is not ported.
- The unused option ELLCONF is dropped.

Remaining sources of difference are floating-point rounding in the mean,
covariance, Cholesky factor and matrix-vector products (Armadillo/LAPACK
versus the plain loops here). They change results at the 1e-16 level, which
only alters a run if a proposal falls within that distance of an accept or
reject boundary. Constraint values computed in Python can likewise differ from
MC++ at the last bit.
