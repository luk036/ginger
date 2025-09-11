import pytest

from ginger.vector2 import Vector2


def test_vector2_init():
    """Test initialization of Vector2."""
    v = Vector2(1.0, 2.0)
    assert v.x == 1.0
    assert v.y == 2.0


def test_vector2_properties():
    """Test properties of Vector2."""
    v = Vector2(1.0, 2.0)
    assert v.x == 1.0
    assert v.y == 2.0


def test_vector2_dot():
    """Test dot product of two Vector2 instances."""
    v1 = Vector2(1.0, 2.0)
    v2 = Vector2(3.0, 4.0)
    assert v1.dot(v2) == 11.0


def test_vector2_sub():
    """Test subtraction of two Vector2 instances."""
    v1 = Vector2(1.0, 2.0)
    v2 = Vector2(3.0, 4.0)
    result = v1 - v2
    assert isinstance(result, Vector2)
    assert result.x == -2.0
    assert result.y == -2.0


def test_vector2_isub():
    """Test in-place subtraction of Vector2."""
    v1 = Vector2(1.0, 2.0)
    v2 = Vector2(3.0, 4.0)
    v1 -= v2
    assert v1.x == -2.0
    assert v1.y == -2.0


def test_vector2_mul():
    """Test scalar multiplication of Vector2."""
    v = Vector2(1.0, 2.0)
    result = v * 2.0
    assert isinstance(result, Vector2)
    assert result.x == 2.0
    assert result.y == 4.0


def test_vector2_imul():
    """Test in-place scalar multiplication of Vector2."""
    v = Vector2(1.0, 2.0)
    v *= 2.0
    assert v.x == 2.0
    assert v.y == 4.0


def test_vector2_truediv():
    """Test scalar division of Vector2."""
    v = Vector2(2.0, 4.0)
    result = v / 2.0
    assert isinstance(result, Vector2)
    assert result.x == 1.0
    assert result.y == 2.0


def test_vector2_truediv_by_zero():
    """Test division by zero."""
    v = Vector2(1.0, 2.0)
    with pytest.raises(ZeroDivisionError):
        v / 0.0


def test_vector2_repr():
    """Test the official string representation of Vector2."""
    v = Vector2(1.0, 2.0)
    assert repr(v) == "Vector2(1.0, 2.0)"


def test_vector2_str():
    """Test the informal string representation of Vector2."""
    v = Vector2(1.0, 2.0)
    assert str(v) == "<1.0, 2.0>"
