from pytest import approx

from ginger.rootfinding import delta, suppress, suppress_old
from ginger.vector2 import Vector2


def test_suppress1():
    vri = Vector2(-2, 0)
    vrj = Vector2(4, -5)

    vA = Vector2(3, 3)
    vA1 = Vector2(1, 2)
    suppress_old(vA, vA1, vri, vrj)
    dr_old = delta(vA, vri, Vector2(vA1._x, vA1._y))

    vA = Vector2(3, 3)
    vA1 = Vector2(1, 2)
    vA, vA1 = suppress(vA, vA1, vri, vrj)
    dr_new = delta(vA, vri, vA1)

    assert dr_new._x == approx(dr_old._x)
    assert dr_new._y == approx(dr_old._y)


def test_suppress2():
    vr0 = Vector2(-2, 0)
    vr1 = Vector2(4, -5)
    vr2 = Vector2(-1, 3)

    vA = Vector2(3, 3)
    vA1 = Vector2(1, 2)
    suppress_old(vA, vA1, vr0, vr1)
    suppress_old(vA, vA1, vr0, vr2)
    dr_old = delta(vA, vr0, Vector2(vA1._x, vA1._y))

    vA = Vector2(3, 3)
    vA1 = Vector2(1, 2)
    suppress_old(vA, vA1, vr0, vr2)
    suppress_old(vA, vA1, vr0, vr1)
    dr_old2 = delta(vA, vr0, Vector2(vA1._x, vA1._y))

    assert dr_old2._x == approx(dr_old._x)
    assert dr_old2._y == approx(dr_old._y)

    vA = Vector2(3, 3)
    vA1 = Vector2(1, 2)
    vA, vA1 = suppress(vA, vA1, vr0, vr1)
    vA, vA1 = suppress(vA, vA1, vr0, vr2)
    dr_new = delta(vA, vr0, vA1)

    vA = Vector2(3, 3)
    vA1 = Vector2(1, 2)
    vA, vA1 = suppress(vA, vA1, vr0, vr2)
    vA, vA1 = suppress(vA, vA1, vr0, vr1)
    dr_new2 = delta(vA, vr0, vA1)

    assert dr_new._x == approx(dr_new2._x)
    assert dr_new._y == approx(dr_new2._y)

    assert dr_new._x == approx(dr_old._x)
    assert dr_new._y == approx(dr_old._y)
