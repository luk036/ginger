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
    """A 2x2 matrix class for linear algebra operations.

    This class represents a 2x2 matrix where each row is a Vector2 object.
    It provides essential matrix operations used in polynomial root-finding
    algorithms, including matrix-vector multiplication and determinant calculation.

    Attributes:
        x (Vector2): The first row of the matrix.
        y (Vector2): The second row of the matrix.

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
        Initialize a 2x2 matrix using two Vector2 rows.

        Parameters:
        x (Vector2): First row vector of the matrix
        y (Vector2): Second row vector of the matrix

        Example:
            >>> m = Matrix2(Vector2(1.0, 2.0), Vector2(3.0, 4.0))
            >>> print(m.x)
            <1.0, 2.0>
            >>> print(m.y)
            <3.0, 4.0>
            >>> print(m.mdot(Vector2(5.0, 6.0)))
            <17.0, 39.0>
            >>> print(m.det())
            -2.0
        """
        self._x = x
        self._y = y

    @property
    def x(self) -> Vector2:
        """
        Get the first row vector of the matrix.

        Returns:
            Vector2: The first row vector

        Examples:
            >>> m = Matrix2(Vector2(1.0, 2.0), Vector2(3.0, 4.0))
            >>> print(m.x)
            <1.0, 2.0>
        """
        return self._x

    @property
    def y(self) -> Vector2:
        """
        Get the second row vector of the matrix.

        Returns:
            Vector2: The second row vector

        Examples:
            >>> m = Matrix2(Vector2(1.0, 2.0), Vector2(3.0, 4.0))
            >>> print(m.y)
            <3.0, 4.0>
        """
        return self._y

    def mdot(self, rhs: Vector2) -> Vector2:
        """
        Matrix-vector multiplication: M * v.

        Parameters:
            rhs (Vector2): Right-hand side vector for multiplication

        Returns:
            Vector2: Result vector of the matrix-vector product

        Calculation:
            [x•rhs]  # Dot product of first row with vector
            [y•rhs]  # Dot product of second row with vector

        Examples:
            >>> m = Matrix2(Vector2(1.0, 2.0), Vector2(3.0, 4.0))
            >>> print(m.mdot(Vector2(5.0, 6.0)))
            <17.0, 39.0>
        """
        return Vector2(self._x.dot(rhs), self._y.dot(rhs))

    def det(self) -> float:
        """
        Calculate the determinant of the 2x2 matrix.

        Formula:
            det = (x.x * y.y) - (x.y * y.x)

        Returns:
            float: Determinant value

        Examples:
            >>> m = Matrix2(Vector2(1.0, 2.0), Vector2(3.0, 4.0))
            >>> print(m.det())
            -2.0
        """
        a11, a12 = self.x.x, self.x.y
        a21, a22 = self.y.x, self.y.y
        return a11 * a22 - a12 * a21

    def __truediv__(self, alpha: float) -> "Matrix2":
        """
        Matrix scalar division: M / α.

        Parameters:
            alpha (float): Scalar divisor (must be non-zero)

        Returns:
            Matrix2: New matrix where each row is divided by alpha

        Operation:
            Returns new Matrix2(x/alpha, y/alpha)
        """
        return Matrix2(self.x / alpha, self.y / alpha)
