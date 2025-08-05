import pytest
from ginger.autocorr import extract_autocorr, initial_autocorr, pbairstow_autocorr
from ginger.rootfinding import Options, find_rootq
from ginger.vector2 import Vector2


def test_extract_autocorr_complex_outside():
    """Test complex roots outside unit circle."""
    # d < 0 (complex roots), |q| > 1
    vr = extract_autocorr(Vector2(1, -4))
    assert vr.x == pytest.approx(0.25)
    assert vr.y == pytest.approx(-0.25)


def test_extract_autocorr_complex_inside():
    """Test complex roots inside unit circle."""
    # d < 0 (complex roots), |q| <= 1
    vr = Vector2(1, -0.5)
    new_vr = extract_autocorr(vr)
    assert new_vr is vr  # Should be unchanged


def test_extract_autocorr_real_one_outside():
    """Test real roots, one outside unit circle."""
    # d > 0, one root outside unit circle
    # roots of x^2 - 3x - 2 = 0 are approx 3.56 and -0.56
    vr = Vector2(3, 2)
    new_vr = extract_autocorr(vr)
    # new roots should be 1/3.56 and -0.56
    roots = find_rootq(new_vr)
    assert abs(roots[0]) <= 1.0
    assert abs(roots[1]) <= 1.0


def test_extract_autocorr_real_two_outside():
    """Test real roots, two outside unit circle."""
    # d > 0, two roots outside unit circle
    # roots are 2, 3
    vr = Vector2(5, -6)
    new_vr = extract_autocorr(vr)
    # new roots should be 1/2, 1/3
    assert new_vr.x == pytest.approx(1 / 2 + 1 / 3)
    assert new_vr.y == pytest.approx(-(1 / 2 * 1 / 3))


def test_extract_autocorr_real_both_inside():
    """Test real roots, both inside unit circle."""
    # d > 0, both roots inside unit circle
    # roots are 0.5, 0.2
    vr = Vector2(0.7, -0.1)
    new_vr = extract_autocorr(vr)
    assert new_vr is vr  # Should be unchanged


def test_autocorr_convergence():
    """Test convergence of pbairstow_autocorr."""
    h = [10.0, 34.0, 75.0, 94.0, 150.0, 94.0, 75.0, 34.0, 10.0]
    vr0s = initial_autocorr(h)
    opts = Options()
    vrs, niter, found = pbairstow_autocorr(h, vr0s, opts)
    print([niter, found])

    for vr in vrs:
        vr = extract_autocorr(vr)
        print(find_rootq(vr))

    assert niter <= 12
    assert found is True


def test_autocorr_non_convergence():
    """Test non-convergence of pbairstow_autocorr."""
    h = [10.0, 34.0, 75.0, 94.0, 150.0, 94.0, 75.0, 34.0, 10.0]
    vr0s = initial_autocorr(h)
    opts = Options()
    opts.max_iters = 1
    vrs, niter, found = pbairstow_autocorr(h, vr0s, opts)
    assert niter == 1
    assert found is False


def test_autocorr_individual_convergence():
    """Test individual convergence within pbairstow_autocorr."""
    h = [10.0, 34.0, 75.0, 94.0, 150.0, 94.0, 75.0, 34.0, 10.0]
    vr0s = initial_autocorr(h)
    # Make one of the initial guesses very good
    vr0s[0] = Vector2(1.4, -0.8)
    opts = Options()
    opts.tol_ind = 1e-1
    vrs, niter, found = pbairstow_autocorr(h, vr0s, opts)
    assert found is True
