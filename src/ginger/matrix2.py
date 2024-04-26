from .vector2 import Vector2

"""
The Matrix2 class represents a 2x2 matrix with two Vector2 objects as its rows.
"""


class Matrix2:
    _x: Vector2
    _y: Vector2

    def __init__(self, x: Vector2, y: Vector2):
        """
        The function initializes an object with two Vector2 parameters.

        :param x: The parameter `x` is of type `Vector2`. It represents a vector in two-dimensional space
        :type x: Vector2
        :param y: The parameter `y` is of type `Vector2`. It represents a vector in two-dimensional space
        :type y: Vector2

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
    def x(self):
        """
        The function returns the value of the private variable `_x`.
        :return: The property `x` is returning the value of the private variable `_x`.

        Examples:
            >>> m = Matrix2(Vector2(1.0, 2.0), Vector2(3.0, 4.0))
            >>> print(m.x)
            <1.0, 2.0>
        """
        return self._x

    @property
    def y(self):
        """
        The function returns the value of the private variable `_y`.
        :return: The property `y` is returning the value of the private variable `_y`.

        Examples:
            >>> m = Matrix2(Vector2(1.0, 2.0), Vector2(3.0, 4.0))
            >>> print(m.y)
            <3.0, 4.0>
        """
        return self._y

    def mdot(self, rhs: Vector2) -> Vector2:
        """
        The `mdot` function performs a matrix-vector product.

        :param rhs: The parameter `rhs` is a Vector2 object that represents the right-hand side vector in the matrix-vector product
        :type rhs: Vector2
        :return: The method `mdot` returns a `Vector2` object.

        Examples:
            >>> m = Matrix2(Vector2(1.0, 2.0), Vector2(3.0, 4.0))
            >>> print(m.mdot(Vector2(5.0, 6.0)))
            <17.0, 39.0>
        """
        return Vector2(self._x.dot(rhs), self._y.dot(rhs))

    def det(self) -> float:
        """
        The `det` function calculates the determinant of a 2x2 matrix.
        :return: The determinant of the matrix.

        Examples:
            >>> m = Matrix2(Vector2(1.0, 2.0), Vector2(3.0, 4.0))
            >>> print(m.det())
            -2.0
        """
        a11, a12 = self.x.x, self.x.y
        a21, a22 = self.y.x, self.y.y
        return a11 * a22 - a12 * a21

    def __truediv__(self, alpha: float):
        """
        The `__truediv__` function divides the x and y components of a Matrix2 object by a scalar value.

        :param alpha: The parameter `alpha` is a scalar value that is used to divide the x and y components of the vector. It is used to scale down the vector by dividing each component by the scalar value
        :type alpha: float
        :return: The `__truediv__` method returns a new `Matrix2` object with the x and y components divided by the given scalar `alpha`.
        """
        return Matrix2(self.x / alpha, self.y / alpha)


if __name__ == "__main__":
    v = Vector2(3.0, 4.0)
    w = Vector2(5.0, 6.0)
    m = Matrix2(v, w)
    print(m.det())
