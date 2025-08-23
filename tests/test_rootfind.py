from ginger.rootfinding import find_rootq, initial_guess, pbairstow_even
from ginger.rootfinding import suppress2, suppress, delta
from ginger.vector2 import Vector2
from pytest import approx


def test_delta1():
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


def test_suppress1():
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


def test_suppress2():
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


def test_suppress3():
    vri = Vector2(-2, 0)
    vrj = Vector2(4, 5)
    vrk = Vector2(3, 7)

    vA = Vector2(3, 3)
    vA1 = Vector2(1, 2)
    vA, vA1 = suppress2(vA, vA1, vri, vrj)
    vA, vA1 = suppress2(vA, vA1, vri, vrk)
    dr1 = delta(vA, vri, vA1)

    vA = Vector2(3, 3)
    vA1 = Vector2(1, 2)
    vA, vA1 = suppress2(vA, vA1, vri, vrk)
    vA, vA1 = suppress2(vA, vA1, vri, vrj)
    dr2 = delta(vA, vri, vA1)
    assert dr1.dot(dr1) == approx(dr2.dot(dr2))


def test_suppress4():
    vri = Vector2(-2, 0)
    vrj = Vector2(4, 5)
    vrk = Vector2(3, 7)
    vrl = Vector2(-3, 1)

    vA = Vector2(3, 3)
    vA1 = Vector2(1, 2)
    vA, vA1 = suppress2(vA, vA1, vri, vrj)
    vA, vA1 = suppress2(vA, vA1, vri, vrk)
    vA, vA1 = suppress2(vA, vA1, vri, vrl)
    dr1 = delta(vA, vri, vA1)

    vA = Vector2(3, 3)
    vA1 = Vector2(1, 2)
    vA, vA1 = suppress2(vA, vA1, vri, vrl)
    vA, vA1 = suppress2(vA, vA1, vri, vrk)
    vA, vA1 = suppress2(vA, vA1, vri, vrj)
    dr2 = delta(vA, vri, vA1)
    assert dr1.dot(dr1) == approx(dr2.dot(dr2))


def test_suppress5():
    vri = Vector2(-2, 0)
    vrj = Vector2(4, 5)
    Vector2(3, 7)
    Vector2(-3, 1)

    vA = Vector2(3, 3)
    vA1 = Vector2(1, 2)
    vA, vA1 = suppress(vA, vA1, vri, vrj)
    # vA, vA1 = suppress(vA, vA1, vri, vrk)
    dr1 = delta(vA, vri, vA1)

    vA = Vector2(3, 3)
    vA1 = Vector2(1, 2)
    vA, vA1 = suppress2(vA, vA1, vri, vrj)
    # vA, vA1 = suppress2(vA, vA1, vri, vrk)
    dr2 = delta(vA, vri, vA1)

    print(dr1)
    print(dr2)
    assert dr1.dot(dr1) == approx(dr2.dot(dr2))


def test_rootfind():
    h = [5.0, 2.0, 9.0, 6.0, 2.0]
    vr0s = initial_guess(h)
    _, niter, found = pbairstow_even(h, vr0s)
    print([niter, found])
    assert niter <= 4


def test_rootfind2():
    h = [10.0, 34.0, 75.0, 94.0, 150.0, 94.0, 75.0, 34.0, 10.0]
    vr0s = initial_guess(h)
    vrs, niter, found = pbairstow_even(h, vr0s)
    print([niter, found])
    print(find_rootq(vr) for vr in vrs)
    assert niter <= 11

def test_find_rootq_negative_hr():
    """Test find_rootq with d >= 0 and hr < 0."""
    # x^2 + 5x + 6 = 0 => r = -5, q = -6
    vr = Vector2(-5, -6)
    roots = find_rootq(vr)
    assert -3.0 == approx(roots[0])
    assert -2.0 == approx(roots[1])
