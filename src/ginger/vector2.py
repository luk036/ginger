class Vector2:
    __slots__ = ("_x", "_y")
    _x: float
    _y: float

    def __init__(self, x, y):
        """
        The function initializes the values of x and y as private variables.

        :param x: The parameter `x` is used to initialize the `_x` attribute of the object. It represents the value of the x-coordinate
        :param y: The parameter `y` is a variable that represents the y-coordinate of a point
        """
        self._x = x
        self._y = y

    @property
    def x(self):
        """
        The function returns the value of the private variable `_x`.
        :return: The property `x` is returning the value of the private variable `_x`.
        """
        return self._x

    @property
    def y(self):
        """
        The function returns the value of the private variable `_y`.
        :return: The method `y` is returning the value of the attribute `_y`.
        """
        return self._y

    def dot(self, rhs):
        """
        The dot function calculates the dot product of two vectors.

        :param rhs: rhs is the right-hand side vector that we want to calculate the dot product with
        :return: The dot product of the two vectors.

        Examples:
            >>> v1 = Vector2(1, 2)
            >>> v2 = Vector2(3, 4)
            >>> v1.dot(v2)
            11
        """
        return self._x * rhs._x + self._y * rhs._y

    def __iadd__(self, rhs):
        """
        The `__iadd__` method adds the x and y components of the right-hand side vector to the x and y
        components of the left-hand side vector and returns the modified left-hand side vector.

        :param rhs: The parameter `rhs` stands for "right-hand side" and represents the object that is being added to the current object. In this case, it is assumed that `rhs` is an instance of the `Vector2` class, which has attributes `x` and `y`.
        :return: The `__iadd__` method returns `self`, which is the updated instance of the `Vector2` class.

        Examples:
            >>> v1 = Vector2(1, 2)
            >>> v2 = Vector2(3, 4)
            >>> v1 += v2
            >>> print(v1)
            <4, 6>
        """
        self._x += rhs.x
        self._y += rhs.y
        return self

    def __add__(self, rhs):
        """
        The function overloads the "+" operator for the Vector2 class to perform vector addition.

        :param rhs: The parameter `rhs` stands for "right-hand side" and represents the vector that is being added to the current vector (`self`)
        :return: The `__add__` method returns a new `Vector2` object that is the result of adding the `x` and `y` components of the current object (`self`) with the `x` and `y` components of the `rhs` object.

        Examples:
            >>> v1 = Vector2(1, 2)
            >>> v2 = Vector2(3, 4)
            >>> print(v1 + v2)
            <4, 6>
            >>> print(v1)
            <1, 2>
        """
        return Vector2(self.x + rhs.x, self.y + rhs.y)

    def __isub__(self, rhs):
        """
        The `__isub__` method subtracts the x and y components of the given vector from the current
        vector and returns the updated vector.

        :param rhs: rhs is the right-hand side operand of the subtraction operation. In this case, it is an instance of the Vector2 class
        :return: The `__isub__` method returns `self` after performing the subtraction operation.

        Examples:
            >>> v1 = Vector2(1, 2)
            >>> v2 = Vector2(3, 4)
            >>> v1 -= v2
            >>> print(v1)
            <-2, -2>
            >>> print(v2)
            <3, 4>
        """
        self._x -= rhs.x
        self._y -= rhs.y
        return self

    def __sub__(self, rhs):
        """
        The function subtracts the x and y components of two Vector2 objects and returns a new Vector2
        object with the result.

        :param rhs: rhs is an instance of the Vector2 class
        :return: The `__sub__` method returns a new `Vector2` object that represents the subtraction of
        the `rhs` vector from the current vector.

        Examples:
            >>> v1 = Vector2(1, 2)
            >>> v2 = Vector2(3, 4)
            >>> print(v1 - v2)
            <-2, -2>
            >>> print(v1)
            <1, 2>
            >>> print(v2)
            <3, 4>
        """
        return Vector2(self.x - rhs.x, self.y - rhs.y)

    def __imul__(self, alpha: float):
        """
        The function multiplies the x and y components of a Vector2 object by a scalar value.

        :param alpha: The parameter "alpha" is a scalar value that is used to multiply the x and y components of the vector
        :type alpha: float
        :return: The method returns the updated instance of the Vector2 object after multiplying its components by the scalar alpha.

        Examples:
            >>> v1 = Vector2(1, 2)
            >>> v1 *= 2
            >>> print(v1)
            <2, 4>
        """
        self._x *= alpha
        self._y *= alpha
        return self

    def __mul__(self, alpha: float):
        """
        The function multiplies a Vector2 object by a scalar value.

        :param alpha: The parameter `alpha` is a scalar value that is used to multiply each component of the vector
        :type alpha: float
        :return: The method returns a new instance of the Vector2 class with the x and y values multiplied by the scalar alpha.

        Examples:
            >>> v1 = Vector2(1, 2)
            >>> print(v1 * 2)
            <2, 4>
            >>> print(v1)
            <1, 2>
        """
        return Vector2(self.x * alpha, self.y * alpha)

    def __itruediv__(self, alpha: float):
        """
        The `__itruediv__` function divides the x and y components of a Vector2 object by a scalar
        value.

        :param alpha: The parameter `alpha` is a scalar value that is used to divide the `x` and `y` components of the vector. It is used to scale down the vector by dividing each component by `alpha`
        :type alpha: float
        :return: The method returns the updated instance of the Vector2 object after performing the division operation.

        Examples:
            >>> v1 = Vector2(1, 2)
            >>> v1 /= 2
            >>> print(v1)
            <0.5, 1.0>
            >>> print(v1)
            <0.5, 1.0>
            >>> v1 /= 0
            Traceback (most recent call last):
                ...
            ZeroDivisionError: float division by zero
            >>> print(v1)
            <0.5, 1.0>
            >>> print(v1)
            <0.5, 1.0>
            >>> v1 /= 1
            >>> print(v1)
            <0.5, 1.0>
            >>> print(v1)
            <0.5, 1.0>
            >>> v1 /= 2
            >>> print(v1)
            <0.25, 0.5>
            >>> print(v1)
            <0.25, 0.5>
            >>> v1 /= 2
            >>> print(v1)
            <0.125, 0.25>
        """
        self._x /= alpha
        self._y /= alpha
        return self

    def __truediv__(self, alpha: float):
        """
        The `__truediv__` function divides the x and y components of a Vector2 object by a scalar value.

        :param alpha: The parameter `alpha` is a scalar value that is used to divide the x and y components of the vector. It is used to scale down the vector by dividing each component by the scalar value
        :type alpha: float
        :return: The `__truediv__` method returns a new `Vector2` object with the x and y components divided by the given scalar `alpha`.

        Examples:
            >>> v1 = Vector2(1, 2)
            >>> print(v1 / 2)
            <0.5, 1.0>
            >>> print(v1)
            <1, 2>
            >>> v1 /= 1
            >>> print(v1)
            <1.0, 2.0>
            >>> v1 /= 2
            >>> print(v1)
            <0.5, 1.0>
            >>> print(v1)
            <0.5, 1.0>
            >>> v1 /= 2
            >>> print(v1)
            <0.25, 0.5>
        """
        return Vector2(self.x / alpha, self.y / alpha)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.x}, {self.y}"

    def __str__(self):
        """
        The __str__ function returns a string representation of a Vector2 object in the format "<x, y>".

        :return: The `__str__` method is returning a formatted string representation of the Vector2 object. The string is in the format "<x, y>", where x and y are the values of the x and y attributes of the Vector2 object.

        Examples:
            >>> v1 = Vector2(1, 2)
            >>> print(v1)
            <1, 2>
            >>> v2 = Vector2(3, 4)
            >>> print(v2)
            <3, 4>
            >>> v3 = Vector2(5, 6)
            >>> print(v3)
            <5, 6>
        """
        # return "<{self.x}, {self.y}>".format(self=self)
        return f"<{self.x}, {self.y}>"


# if __name__ == "__main__":
#     v = Vector2(3.0, 4.0)
#     w = Vector2(5.0, 6.0)
#     print(v.dot(w))
