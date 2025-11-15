class Vector2:
    """A 2D vector class for mathematical vector operations.

    This class represents a two-dimensional vector with x and y components.
    It provides basic vector operations like addition, subtraction, scalar
    multiplication, dot product, and string representation. The class uses
    slots for memory efficiency and implements proper operator overloading.
    """

    __slots__ = ("_x", "_y")
    _x: float
    _y: float

    def __init__(self, x: float, y: float) -> None:
        """
        Initialize a new Vector2 instance with x and y components.

        This constructor creates a new 2D vector with the specified x and y
        components. The components are stored as private attributes (_x and _y)
        to enforce encapsulation.

        :param x: The x-component of the vector (can be int or float)
        :param y: The y-component of the vector (can be int or float)

        Examples:
            >>> v = Vector2(3.0, 4.0)
            >>> print(v)
            <3.0, 4.0>
        """
        self._x = x
        self._y = y

    @property
    def x(self) -> float:
        """
        Getter property for the x-component of the vector.

        This property provides read-only access to the private _x attribute,
        maintaining encapsulation while allowing external access to the value.

        :return: The x-component of the vector as a float
        :rtype: float

        Examples:
            >>> v = Vector2(3.0, 4.0)
            >>> v.x
            3.0
        """
        return self._x

    @property
    def y(self) -> float:
        """
        Getter property for the y-component of the vector.

        This property provides read-only access to the private _y attribute,
        maintaining encapsulation while allowing external access to the value.

        :return: The y-component of the vector as a float
        :rtype: float

        Examples:
            >>> v = Vector2(3.0, 4.0)
            >>> v.y
            4.0
        """
        return self._y

    def dot(self, rhs: "Vector2") -> float:
        """
        Calculate the dot product of this vector with another vector.

        The dot product (also called scalar product) is a fundamental operation
        in vector mathematics that returns a single scalar value representing
        the magnitude of the projection of one vector onto another.

        :param rhs: The right-hand side vector for the dot product operation
        :type rhs: Vector2
        :return: The scalar dot product result
        :rtype: float

        Examples:
            >>> v1 = Vector2(1, 2)
            >>> v2 = Vector2(3, 4)
            >>> v1.dot(v2)
            11
        """
        return self._x * rhs._x + self._y * rhs._y

    def __isub__(self, rhs: "Vector2") -> "Vector2":
        """
        In-place vector subtraction (-= operator).

        This method modifies the current vector by subtracting another vector's
        components from it. It implements the -= operator for Vector2 objects.

        :param rhs: The vector to subtract from this one
        :type rhs: Vector2
        :return: The modified vector (self)
        :rtype: Vector2

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

    def __sub__(self, rhs: "Vector2") -> "Vector2":
        """
        Vector subtraction (- operator).

        This method creates a new vector that is the difference between this
        vector and another vector. It implements the - operator for Vector2 objects.

        :param rhs: The vector to subtract from this one
        :type rhs: Vector2
        :return: A new vector representing the difference
        :rtype: Vector2

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

    def __imul__(self, alpha: float) -> "Vector2":
        """
        In-place scalar multiplication (*= operator).

        This method modifies the current vector by multiplying its components
        by a scalar value. It implements the *= operator for Vector2 objects.

        :param alpha: The scalar multiplier
        :type alpha: float
        :return: The modified vector (self)
        :rtype: Vector2

        Examples:
            >>> v1 = Vector2(1, 2)
            >>> v1 *= 2
            >>> print(v1)
            <2, 4>
        """
        self._x *= alpha
        self._y *= alpha
        return self

    def __mul__(self, alpha: float) -> "Vector2":
        """
        Scalar multiplication (* operator).

        This method creates a new vector that is the result of multiplying this
        vector's components by a scalar value. It implements the * operator for
        Vector2 objects.

        :param alpha: The scalar multiplier
        :type alpha: float
        :return: A new scaled vector
        :rtype: Vector2

        Examples:
            >>> v1 = Vector2(1, 2)
            >>> print(v1 * 2)
            <2, 4>
            >>> print(v1)
            <1, 2>
        """
        return Vector2(self.x * alpha, self.y * alpha)

    def __truediv__(self, alpha: float) -> "Vector2":
        """
        Scalar division (/ operator).

        This method creates a new vector that is the result of dividing this
        vector's components by a scalar value. It implements the / operator for
        Vector2 objects.

        :param alpha: The scalar divisor (must not be zero)
        :type alpha: float
        :return: A new scaled vector
        :rtype: Vector2

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

    def __repr__(self) -> str:
        """Official string representation of the Vector2 object.

        This method provides the official string representation of the Vector2
        object, which can be used to recreate the object using eval().

        :return: A string representation that can recreate the object
        :rtype: str
        """
        return f"{self.__class__.__name__}({self.x}, {self.y})"

    def __str__(self) -> str:
        """
        Informal string representation of the Vector2 object.

        This method provides a human-readable string representation of the
        vector in the format "<x, y>". It implements the str() conversion.

        :return: A formatted string showing the vector components
        :rtype: str

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
