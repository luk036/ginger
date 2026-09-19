"""Degree-scaling benchmarks for the solvers.

Records iteration counts alongside wall-clock time. These solvers cost
``O(niter * n^2)``, so a regression that silently pushes ``niter`` to
``max_iters`` shows up as an orders-of-magnitude slowdown with no change to
the per-iteration work -- invisible to wall-clock-only benchmarks.

Run with::

    pytest bench/ --benchmark-only
"""

import random
from typing import Any, List, Tuple

import pytest

from ginger.aberth import aberth, initial_aberth, leja_order, poly_from_roots
from ginger.rootfinding import Options, initial_guess, pbairstow_even

DEGREES = [8, 16, 32]


def real_poly(degree: int, seed: int = 42) -> List[float]:
    """Real-coefficient monic polynomial whose roots have modulus ~3."""
    rng = random.Random(seed)
    roots: List[complex] = []
    while len(roots) < degree:
        re = rng.uniform(-3.0, 3.0)
        im = rng.uniform(0.2, 3.0)
        roots.append(complex(re, im))
        if len(roots) < degree:
            roots.append(complex(re, -im))
    return poly_from_roots(roots)


@pytest.mark.parametrize("degree", DEGREES)
def test_aberth_scaling(benchmark: Any, degree: int) -> None:
    coeffs = real_poly(degree)
    zs = initial_aberth(coeffs)

    def run() -> Tuple[int, bool]:
        _, niter, found = aberth(coeffs, list(zs), Options())
        return niter, found

    niter, found = benchmark(run)
    assert niter <= Options.max_iters
    if degree == 16:
        assert found
        assert niter < 20


@pytest.mark.parametrize("degree", DEGREES)
def test_pbairstow_scaling(benchmark: Any, degree: int) -> None:
    coeffs = real_poly(degree)
    vrs = initial_guess(coeffs)

    def run() -> int:
        _, niter, _ = pbairstow_even(coeffs, list(vrs), Options())
        return niter

    niter = benchmark(run)
    assert niter <= Options.max_iters


@pytest.mark.parametrize("n", [50, 100, 200])
def test_leja_order_scaling(benchmark: Any, n: int) -> None:
    rng = random.Random(42)
    points = [complex(rng.random(), rng.random()) for _ in range(n)]
    result = benchmark(leja_order, points)
    assert len(result) == n
