import pytest

from ginger.aberth import (
    aberth,
    aberth_autocorr,
    aberth_autocorr_mt,
    aberth_mt,
    initial_aberth,
    initial_aberth_autocorr,
    initial_aberth_autocorr_orig,
    initial_aberth_orig,
    leja_order,
    poly_from_autocorr_roots,
    poly_from_roots,
)
from ginger.rootfinding import Options


def test_aberth1() -> None:
    h = [5.0, 2.0, 9.0, 6.0, 2.0]

    z0s = initial_aberth(h)
    _, niter, found = aberth(h, z0s)
    print([niter, found])
    assert niter <= 7

    z0s = initial_aberth_orig(h)
    _, niter_orig, found = aberth(h, z0s)
    print([niter_orig, found])
    assert niter <= niter_orig

    z0s = initial_aberth(h)
    _, niter, found = aberth_mt(h, z0s)
    print([niter, found])
    assert niter <= 7


def test_aberth2() -> None:
    h = [10.0, 34.0, 75.0, 94.0, 150.0, 94.0, 75.0, 34.0, 10.0]

    z0s = initial_aberth(h)
    zs, niter, found = aberth(h, z0s)
    print([niter, found])
    print([z for z in zs])
    assert niter <= 8

    z0s = initial_aberth_orig(h)
    _, niter_orig, found = aberth(h, z0s)
    print([niter_orig, found])
    assert niter_orig <= 8

    z0s = initial_aberth(h)
    zs, niter, found = aberth_mt(h, z0s)
    print([niter, found])
    print([z for z in zs])
    assert niter <= 9

    z0s = initial_aberth_autocorr(h)
    zs, niter, found = aberth_autocorr_mt(h, z0s)
    print([niter, found])
    print([z for z in zs])
    assert niter <= 10


def test_initial_aberth_autocorr_large_radius() -> None:
    """Test initial_aberth_autocorr with a large radius."""
    h = [1.0, 0.0, 100.0]
    z0s = initial_aberth_autocorr(h)
    for z in z0s:
        assert abs(z) < 1.1


def test_aberth_non_convergence() -> None:
    """Test non-convergence of aberth."""
    h = [5.0, 2.0, 9.0, 6.0, 2.0]
    z0s = initial_aberth(h)
    opts = Options()
    opts.max_iters = 1
    _, _, found = aberth(h, z0s, opts)
    assert found is False


def test_aberth_autocorr_single_root() -> None:
    """Test aberth_autocorr with a single root."""
    h = [1.0, -1.0]
    z0s = [0.0 + 0j]  # Use complex number to match expected type
    zs, niter, found = aberth_autocorr(h, z0s)
    assert found is True


r = [
    -0.00196191,
    -0.00094597,
    -0.00023823,
    0.00134667,
    0.00380494,
    0.00681596,
    0.0097864,
    0.01186197,
    0.0121238,
    0.00985211,
    0.00474894,
    -0.00281751,
    -0.01173923,
    -0.0201885,
    -0.02590168,
    -0.02658216,
    -0.02035729,
    -0.00628271,
    0.01534627,
    0.04279982,
    0.0732094,
    0.10275561,
    0.12753013,
    0.14399228,
    0.15265722,
    0.14399228,
    0.12753013,
    0.10275561,
    0.0732094,
    0.04279982,
    0.01534627,
    -0.00628271,
    -0.02035729,
    -0.02658216,
    -0.02590168,
    -0.0201885,
    -0.01173923,
    -0.00281751,
    0.00474894,
    0.00985211,
    0.0121238,
    0.01186197,
    0.0097864,
    0.00681596,
    0.00380494,
    0.00134667,
    -0.00023823,
    -0.00094597,
    -0.00196191,
]


def test_aberth_fir() -> None:
    opt = Options()
    opt.tolerance = 1e-8

    z0s = initial_aberth(r)
    zs, niter, found = aberth(r, z0s, opt)
    print([niter, found])
    for z in zs:
        print(z)
    assert niter <= 12

    z0s = initial_aberth_orig(r)
    zs, niter_orig, found = aberth(r, z0s, opt)
    print([niter, found])
    for z in zs:
        print(z)
    assert niter <= 12
    assert niter <= niter_orig

    z0s = initial_aberth(r)
    zs, niter, found = aberth_mt(r, z0s, opt)
    print([niter, found])
    for z in zs:
        print(z)
    assert niter <= 18


def test_aberth_autocorr_fir() -> None:
    opt = Options()
    opt.tolerance = 1e-13

    z0s = initial_aberth_autocorr(r)
    zs, niter, found = aberth_autocorr(r, z0s, opt)
    print([niter, found])
    for z in zs:
        print(z)
    assert niter <= 11

    z0s = initial_aberth_autocorr_orig(r)
    zs, niter_orig, found = aberth_autocorr(r, z0s, opt)
    print([niter_orig, found])
    for z in zs:
        print(z)
    assert niter_orig <= 11
    # assert niter <= niter_orig

    z0s = initial_aberth_autocorr(r)
    zs, niter, found = aberth_autocorr_mt(r, z0s, opt)
    print([niter, found])
    for z in zs:
        print(z)
    assert niter <= 15


def test_leja_order_empty() -> None:
    assert leja_order([]) == []


def test_leja_order_single() -> None:
    ordered = leja_order([3.0 + 4.0j])
    assert ordered == [3.0 + 4.0j]


def test_leja_order_preserves_set() -> None:
    pts = [3.0 + 1.0j, 1.0 + 2.0j, 0.5 + 0.5j, -2.0 - 1.0j]
    ordered = leja_order(pts)
    assert len(ordered) == len(pts)
    assert set(ordered) == set(pts)


def test_leja_order_smallest_first() -> None:
    pts = [10.0 + 0.0j, 0.5 + 0.0j, 3.0 + 0.0j]
    ordered = leja_order(pts)
    assert ordered[0] == 0.5 + 0.0j


def test_poly_from_roots_known() -> None:
    coeffs = poly_from_roots([1.0 + 0.0j, -1.0 + 0.0j])
    assert coeffs == [1.0, 0.0, -1.0]


def test_poly_from_roots_complex_conj() -> None:
    coeffs = poly_from_roots([0.0 + 1.0j, 0.0 - 1.0j])
    assert coeffs == [1.0, 0.0, 1.0]


def test_poly_from_roots_triple() -> None:
    coeffs = poly_from_roots([0.0 + 0.0j, 1.0 + 0.0j, -1.0 + 0.0j])
    assert coeffs == [1.0, 0.0, -1.0, 0.0]


def test_poly_from_roots_empty() -> None:
    assert poly_from_roots([]) == [1.0]


def test_poly_from_roots_aberth_reconstruction() -> None:
    h = [10.0, 34.0, 75.0, 94.0, 150.0, 94.0, 75.0, 34.0, 10.0]
    zs = initial_aberth(h)
    opt = Options()
    opt.tolerance = 1e-12
    zs, niter, found = aberth(h, zs, opt)
    assert found
    monic = poly_from_roots(zs)
    assert len(monic) == len(h)
    scale = h[0]
    for i in range(len(h)):
        assert monic[i] * scale == pytest.approx(h[i], abs=1e-8)


def test_poly_from_autocorr_roots_empty() -> None:
    assert poly_from_autocorr_roots([]) == [1.0]


def test_poly_from_autocorr_roots_reconstruction() -> None:
    h = [10.0, 34.0, 75.0, 94.0, 150.0, 94.0, 75.0, 34.0, 10.0]
    zs = initial_aberth_autocorr(h)
    opt = Options()
    opt.tolerance = 1e-12
    zs, niter, found = aberth_autocorr(h, zs, opt)
    assert found
    monic = poly_from_autocorr_roots(zs)
    assert len(monic) == len(h)
    scale = h[0]
    for i in range(len(h)):
        assert monic[i] * scale == pytest.approx(h[i], abs=1e-8)
