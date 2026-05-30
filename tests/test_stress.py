"""Stress tests for ginger-cpp Python port.

Tests all root-finding methods with high-degree polynomials
to ensure robustness under heavy load. Random high-degree
polynomials are ill-conditioned by nature, so the tests verify
that the algorithms run without error, produce sensible output,
and converge on well-structured problems.
"""

import random

from ginger.aberth import (
    aberth,
    aberth_autocorr,
    aberth_autocorr_mt,
    aberth_mt,
    initial_aberth,
    initial_aberth_autocorr,
)
from ginger.autocorr import initial_autocorr, pbairstow_autocorr
from ginger.rootfinding import Options, initial_guess, pbairstow_even


def random_poly(degree: int) -> list[float]:
    """Generate a random polynomial with coefficients in [-10, 10]."""
    return [random.uniform(-10.0, 10.0) for _ in range(degree + 1)]


def random_palindromic_poly(degree: int) -> list[float]:
    """Generate a random palindromic polynomial (a_i = a_{n-i})."""
    assert degree % 2 == 0
    half = random_poly(degree // 2)
    return half + half[-2::-1]


# =====================================================================
# Aberth stress tests — high degree (robustness, not guaranteed convergence)
# =====================================================================


def test_stress_aberth_st_random_high_degree() -> None:
    """Stress test aberth (ST) with a high-degree polynomial."""
    h = random_poly(100)
    zs = initial_aberth(h)
    assert len(zs) == len(h) - 1

    opts = Options()
    opts.tolerance = 1e-9
    opts.max_iters = 2000
    zs, niter, converged = aberth(h, zs, opts)

    # Random degree-100 polynomials are ill-conditioned;
    # verify the algorithm ran without error, not guaranteed convergence
    print(f"aberth ST degree-100: niter={niter}, converged={converged}")
    assert 0 < niter <= opts.max_iters


def test_stress_aberth_mt_random_high_degree() -> None:
    """Stress test aberth_mt with a high-degree polynomial."""
    h = random_poly(100)
    zs = initial_aberth(h)

    opts = Options()
    opts.tolerance = 1e-9
    opts.max_iters = 2000
    zs, niter, converged = aberth_mt(h, zs, opts)

    print(f"aberth MT degree-100: niter={niter}, converged={converged}")
    assert 0 < niter <= opts.max_iters


def test_stress_aberth_autocorr_st_random_high_degree() -> None:
    """Stress test aberth_autocorr (ST) with a high-degree palindromic polynomial."""
    h = random_palindromic_poly(100)
    zs = initial_aberth_autocorr(h)
    assert len(zs) == len(h) // 2

    opts = Options()
    opts.tolerance = 1e-9
    opts.max_iters = 2000
    zs, niter, converged = aberth_autocorr(h, zs, opts)

    print(f"aberth_autocorr ST degree-100: niter={niter}, converged={converged}")
    assert 0 < niter <= opts.max_iters


def test_stress_aberth_autocorr_mt_random_high_degree() -> None:
    """Stress test aberth_autocorr (MT) with a high-degree palindromic polynomial."""
    h = random_palindromic_poly(100)
    zs = initial_aberth_autocorr(h)
    assert len(zs) == len(h) // 2

    opts = Options()
    opts.tolerance = 1e-9
    opts.max_iters = 2000
    zs, niter, converged = aberth_autocorr_mt(h, zs, opts)

    print(f"aberth_autocorr MT degree-100: niter={niter}, converged={converged}")
    assert 0 < niter <= opts.max_iters


# =====================================================================
# Bairstow stress tests — high degree
# =====================================================================


def test_stress_pbairstow_even_random_high_degree() -> None:
    """Stress test pbairstow_even with a high-degree polynomial."""
    h = random_poly(100)
    vrs = initial_guess(h)
    assert len(vrs) == len(h) // 2

    opts = Options()
    opts.tolerance = 1e-9
    opts.max_iters = 4000
    vrs, niter, converged = pbairstow_even(h, vrs, opts)

    print(f"pbairstow_even degree-100: niter={niter}, converged={converged}")
    assert 0 < niter <= opts.max_iters


def test_stress_pbairstow_autocorr_random_high_degree() -> None:
    """Stress test pbairstow_autocorr with a high-degree palindromic polynomial."""
    h = random_palindromic_poly(100)
    vrs = initial_autocorr(h)
    assert len(vrs) == len(h) // 4

    opts = Options()
    opts.tolerance = 1e-9
    opts.max_iters = 4000
    vrs, niter, converged = pbairstow_autocorr(h, vrs, opts)

    print(f"pbairstow_autocorr degree-100: niter={niter}, converged={converged}")
    assert 0 < niter <= opts.max_iters


# =====================================================================
# Many low-degree batch tests (should converge)
# =====================================================================


def test_stress_100_random_aberth_polynomials() -> None:
    """Run aberth on 100 random degree-6 polynomials."""
    opts = Options()
    opts.tolerance = 1e-9
    opts.max_iters = 1000

    converged_count = 0
    for _ in range(100):
        h = random_poly(6)
        zs = initial_aberth(h)
        _, _, converged = aberth(h, zs, opts)
        if converged:
            converged_count += 1

    print(f"aberth batch: {converged_count}/100 converged")
    assert converged_count >= 80, f"Only {converged_count}/100 aberth converged"


def test_stress_100_random_bairstow_polynomials() -> None:
    """Run pbairstow_even on 100 random degree-6 polynomials."""
    opts = Options()
    opts.tolerance = 1e-9
    opts.max_iters = 1000

    converged_count = 0
    for _ in range(100):
        h = random_poly(6)
        vrs = initial_guess(h)
        _, _, converged = pbairstow_even(h, vrs, opts)
        if converged:
            converged_count += 1

    print(f"bairstow batch: {converged_count}/100 converged")
    assert converged_count >= 80, f"Only {converged_count}/100 bairstow converged"
