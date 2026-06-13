"""2x2 matrix operations for polynomial root-finding algorithms.

This module provides the Matrix2 class which represents a 2x2 matrix with two
Vector2 objects as rows. It is primarily used in Bairstow's method for computing
the adjustments to root estimates via matrix-vector operations and determinant
calculations.

The matrix is stored in row-major order:
    [[x.x, x.y],
     [y.x, y.y]]

Example:
    >>> from ginger.vector2 import Vector2
    >>> m = Matrix2(Vector2(1.0, 2.0), Vector2(3.0, 4.0))
    >>> print(m.det())
    -2.0
"""

from .vector2 import Vector2


class Matrix2:
    """A 2x2 matrix used in Bairstow's correction step.

    Each row is a Vector2. Provides matrix-vector multiplication (:meth:`mdot`)
    and determinant (:meth:`det`) for solving the 2x2 linear system in
    root estimate refinement.

    Attributes:
        x (Vector2): The first row
        y (Vector2): The second row

    Example:
        >>> m = Matrix2(Vector2(1.0, 2.0), Vector2(3.0, 4.0))
        >>> print(m.mdot(Vector2(5.0, 6.0)))
        <17.0, 39.0>
        >>> print(m.det())
        -2.0
    """

    _x: Vector2
    _y: Vector2

    def __init__(self, x: Vector2, y: Vector2):
        """
        Initialize the 2x2 matrix from two row vectors.

        :param x: First row
        :param y: Second row

        Example:
            >>> m = Matrix2(Vector2(1.0, 2.0), Vector2(3.0, 4.0))
            >>> print(m.x)
            <1.0, 2.0>
        """
        self._x = x
        self._y = y

    @property
    def x(self) -> Vector2:
        """First row of the matrix.

        Examples:
            >>> m = Matrix2(Vector2(1.0, 2.0), Vector2(3.0, 4.0))
            >>> print(m.x)
            <1.0, 2.0>
        """
        return self._x

    @property
    def y(self) -> Vector2:
        """Second row of the matrix.

        Examples:
            >>> m = Matrix2(Vector2(1.0, 2.0), Vector2(3.0, 4.0))
            >>> print(m.y)
            <3.0, 4.0>
        """
        return self._y

    def mdot(self, rhs: Vector2) -> Vector2:
        """Matrix-vector multiplication: M * v.

        Returns Vector2(x·rhs, y·rhs).

        :param rhs: Right-hand side vector
        :return: Result of matrix-vector product

        Examples:
            >>> m = Matrix2(Vector2(1.0, 2.0), Vector2(3.0, 4.0))
            >>> print(m.mdot(Vector2(5.0, 6.0)))
            <17.0, 39.0>
        """
        return Vector2(self._x.dot(rhs), self._y.dot(rhs))

    def det(self) -> float:
        """Calculate the determinant: x.x·y.y - x.y·y.x.

        Examples:
            >>> m = Matrix2(Vector2(1.0, 2.0), Vector2(3.0, 4.0))
            >>> print(m.det())
            -2.0
        """
        a11, a12 = self.x.x, self.x.y
        a21, a22 = self.y.x, self.y.y
        return a11 * a22 - a12 * a21

    def __truediv__(self, alpha: float) -> "Matrix2":
        """Scalar division: M / α.

        :param alpha: Divisor (non-zero)
        :return: New matrix with each row divided by alpha
        """
        return Matrix2(self.x / alpha, self.y / alpha)
