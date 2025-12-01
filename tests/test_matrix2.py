import pytest

from ginger.matrix2 import Matrix2
from ginger.vector2 import Vector2


def test_matrix2_init() -> None:
    """Test initialization of Matrix2."""
    x = Vector2(1.0, 2.0)
    y = Vector2(3.0, 4.0)
    m = Matrix2(x, y)
    assert m.x is x
    assert m.y is y


def test_matrix2_properties() -> None:
    """Test properties of Matrix2."""
    x = Vector2(1.0, 2.0)
    y = Vector2(3.0, 4.0)
    m = Matrix2(x, y)
    assert m.x.x == 1.0
    assert m.x.y == 2.0
    assert m.y.x == 3.0
    assert m.y.y == 4.0


def test_matrix2_mdot() -> None:
    """Test matrix-vector multiplication."""
    m = Matrix2(Vector2(1.0, 2.0), Vector2(3.0, 4.0))
    v = Vector2(5.0, 6.0)
    result = m.mdot(v)
    assert isinstance(result, Vector2)
    assert result.x == 17.0
    assert result.y == 39.0


def test_matrix2_det() -> None:
    """Test determinant calculation."""
    m = Matrix2(Vector2(1.0, 2.0), Vector2(3.0, 4.0))
    assert m.det() == -2.0


def test_matrix2_truediv() -> None:
    """Test scalar division of Matrix2."""
    m = Matrix2(Vector2(2.0, 4.0), Vector2(6.0, 8.0))
    result = m / 2.0
    assert isinstance(result, Matrix2)
    assert result.x.x == 1.0
    assert result.x.y == 2.0
    assert result.y.x == 3.0
    assert result.y.y == 4.0


def test_matrix2_truediv_by_zero() -> None:
    """Test division by zero."""
    m = Matrix2(Vector2(1.0, 2.0), Vector2(3.0, 4.0))
    with pytest.raises(ZeroDivisionError):
        m / 0.0
