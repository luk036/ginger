from pytest import approx

from ginger.rootfinding import (
    Options,
    delta,
    find_rootq,
    initial_guess,
    pbairstow_even,
    poly_from_quadratic_factors,
    roots_from_quadratic,
    suppress,
)
from ginger.vector2 import Vector2


def test_delta1() -> None:
    vri = Vector2(-2, 0)
    vrj = Vector2(4, 5)
    vrk = Vector2(3, 7)
    vpj = vri - vrj
    vpk = vri - vrk

    vA = Vector2(3, 3)
    vA = delta(vA, vri, vpj)
    dr1 = delta(vA, vri, vpk)

    vA = Vector2(3, 3)
    vA = delta(vA, vri, vpk)
    dr2 = delta(vA, vri, vpj)
    assert dr1.dot(dr1) == approx(dr2.dot(dr2))


def test_suppress1() -> None:
    vri = Vector2(-2, 0)
    vrj = Vector2(4, 5)
    vrk = Vector2(3, 7)

    vA = Vector2(3, 3)
    vA1 = Vector2(1, 2)
    vA, vA1 = suppress(vA, vA1, vri, vrj)
    vA, vA1 = suppress(vA, vA1, vri, vrk)
    dr1 = delta(vA, vri, vA1)

    vA = Vector2(3, 3)
    vA1 = Vector2(1, 2)
    vA, vA1 = suppress(vA, vA1, vri, vrk)
    vA, vA1 = suppress(vA, vA1, vri, vrj)
    dr2 = delta(vA, vri, vA1)
    assert dr1.dot(dr1) == approx(dr2.dot(dr2))


def test_suppress2() -> None:
    vri = Vector2(-2, 0)
    vrj = Vector2(4, 5)
    vrk = Vector2(3, 7)
    vrl = Vector2(-3, 1)

    vA = Vector2(3, 3)
    vA1 = Vector2(1, 2)
    vA, vA1 = suppress(vA, vA1, vri, vrj)
    vA, vA1 = suppress(vA, vA1, vri, vrk)
    vA, vA1 = suppress(vA, vA1, vri, vrl)
    dr1 = delta(vA, vri, vA1)

    vA = Vector2(3, 3)
    vA1 = Vector2(1, 2)
    vA, vA1 = suppress(vA, vA1, vri, vrl)
    vA, vA1 = suppress(vA, vA1, vri, vrk)
    vA, vA1 = suppress(vA, vA1, vri, vrj)
    dr2 = delta(vA, vri, vA1)
    assert dr1.dot(dr1) == approx(dr2.dot(dr2))


def test_rootfind() -> None:
    h = [5.0, 2.0, 9.0, 6.0, 2.0]
    vr0s = initial_guess(h)
    _, niter, found = pbairstow_even(h, vr0s)
    print([niter, found])
    assert niter <= 4


def test_rootfind2() -> None:
    h = [10.0, 34.0, 75.0, 94.0, 150.0, 94.0, 75.0, 34.0, 10.0]
    vr0s = initial_guess(h)
    vrs, niter, found = pbairstow_even(h, vr0s)
    print([niter, found])
    print(find_rootq(vr) for vr in vrs)
    assert niter <= 11


def test_find_rootq_negative_hr() -> None:
    """Test find_rootq with d >= 0 and hr < 0."""
    # x^2 + 5x + 6 = 0 => r = -5, q = -6
    vr = Vector2(-5, -6)
    roots = find_rootq(vr)
    assert -3.0 == approx(roots[0])
    assert -2.0 == approx(roots[1])


def test_roots_from_quadratic_real() -> None:
    r1, r2 = roots_from_quadratic(Vector2(0.0, 1.0))  # x^2 - 1
    assert abs(r1 - 1.0) < 1e-14
    assert abs(r2 + 1.0) < 1e-14


def test_roots_from_quadratic_complex() -> None:
    r1, r2 = roots_from_quadratic(Vector2(0.0, -1.0))  # x^2 + 1
    assert abs(r1 - 1j) < 1e-14
    assert abs(r2 + 1j) < 1e-14


def test_poly_from_quadratic_factors_empty() -> None:
    assert poly_from_quadratic_factors([]) == [1.0]


def test_poly_from_quadratic_factors_single() -> None:
    # x^2 - 1
    coeffs = poly_from_quadratic_factors([Vector2(0.0, 1.0)])
    assert coeffs == [1.0, 0.0, -1.0]


def test_poly_from_quadratic_factors_two() -> None:
    # (x^2 - 1)(x^2 - 4) = x^4 - 5x^2 + 4
    coeffs = poly_from_quadratic_factors([Vector2(0.0, 1.0), Vector2(0.0, 4.0)])
    assert coeffs[0] == approx(1.0)
    assert coeffs[1] == approx(0.0)
    assert coeffs[2] == approx(-5.0)
    assert coeffs[3] == approx(0.0)
    assert coeffs[4] == approx(4.0)


def test_poly_from_quadratic_factors_general() -> None:
    # (x^2 - 3x - 10)(x^2 + x - 2) = x^4 - 2x^3 - 15x^2 - 4x + 20
    coeffs = poly_from_quadratic_factors([Vector2(3.0, 10.0), Vector2(-1.0, 2.0)])
    assert coeffs[0] == approx(1.0)
    assert coeffs[1] == approx(-2.0)
    assert coeffs[2] == approx(-15.0)
    assert coeffs[3] == approx(-4.0)
    assert coeffs[4] == approx(20.0)


def test_poly_from_quadratic_factors_reconstruction() -> None:
    h = [10.0, 34.0, 75.0, 94.0, 150.0, 94.0, 75.0, 34.0, 10.0]
    vrs = initial_guess(h)
    opt = Options()
    opt.tolerance = 1e-12
    vrs, niter, found = pbairstow_even(h, vrs, opt)
    assert found
    monic = poly_from_quadratic_factors(vrs)
    assert len(monic) == len(h)
    scale = h[0]
    for i in range(len(h)):
        assert monic[i] * scale == approx(h[i], abs=1e-8)
