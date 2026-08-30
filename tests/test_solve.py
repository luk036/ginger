"""Tests for the solve() facade entry points (auto policy selection)."""

from ginger.rootfinding import Options, initial_guess, should_parallelize
from ginger.solve import SolveMode, solve_aberth, solve_aberth_autocorr, solve_pbairstow_even


def test_should_parallelize_threshold() -> None:
    assert not should_parallelize(0)
    assert not should_parallelize(4)
    assert should_parallelize(5)


def test_solve_aberth_automatic_small() -> None:
    """4 roots -> automatic dispatches to the single-threaded variant."""
    h = [10.0, 34.0, 75.0, 94.0, 150.0, 94.0, 75.0, 34.0, 10.0]
    options = Options()
    options.tolerance = 1e-12

    from ginger.aberth import initial_aberth, poly_from_roots

    zs = initial_aberth(h)
    zs, niter, found = solve_aberth(h, zs, options)
    assert found
    monic = poly_from_roots(zs)
    assert len(monic) == len(h)
    scale = h[0]
    for i in range(len(h)):
        assert abs(monic[i] * scale - h[i]) < 1e-8


def test_solve_aberth_explicit_modes() -> None:
    h = [1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 3.0, 0.0, 2.0, 0.0, 1.0]
    options = Options()
    options.tolerance = 1e-12

    from ginger.aberth import initial_aberth

    zs = initial_aberth(h)
    _, _, found = solve_aberth(h, zs, options, SolveMode.SEQUENTIAL)
    assert found
    zs = initial_aberth(h)
    _, _, found = solve_aberth(h, zs, options, SolveMode.MULTI_THREADED)
    assert found


def test_solve_pbairstow_even() -> None:
    h = [10.0, 34.0, 75.0, 94.0, 150.0, 94.0, 75.0, 34.0, 10.0]
    options = Options()
    options.tolerance = 1e-12

    from ginger.aberth import poly_from_roots
    from ginger.rootfinding import roots_from_quadratic

    vrs = initial_guess(h)
    vrs, niter, found = solve_pbairstow_even(h, vrs, options)
    assert found
    all_roots = []
    for vr in vrs:
        r1, r2 = roots_from_quadratic(vr)
        all_roots.extend([r1, r2])
    monic = poly_from_roots(all_roots)
    assert len(monic) == len(h)
    scale = h[0]
    for i in range(len(h)):
        assert abs(monic[i] * scale - h[i]) < 1e-8


def test_solve_aberth_autocorr() -> None:
    h = [10.0, 34.0, 75.0, 94.0, 150.0, 94.0, 75.0, 34.0, 10.0]
    options = Options()
    options.tolerance = 1e-12

    from ginger.aberth import initial_aberth_autocorr, poly_from_autocorr_roots

    zs = initial_aberth_autocorr(h)
    zs, niter, found = solve_aberth_autocorr(h, zs, options)
    assert found
    monic = poly_from_autocorr_roots(zs)
    assert len(monic) == len(h)
    scale = h[0]
    for i in range(len(h)):
        assert abs(monic[i] * scale - h[i]) < 1e-8
