from functools import reduce
from itertools import accumulate
from math import cos, pi, sqrt
from typing import Any, List, Tuple, Union

from lds_gen.lds import VdCorput
from mywheel.robin import Robin

from .matrix2 import Matrix2
from .vector2 import Vector2

Num = Union[float, complex]

PI = pi


# The class "Options" defines default values for maximum iterations, tolerance, and individual
# tolerance.
class Options:
    max_iters: int = 2000
    tolerance: float = 1e-12
    tol_ind: float = 1e-15
    # tol_suppress: float = 1e-1


#                      -1
#    ⎛r ⋅ p + s     p⎞     ⎛A⎞
#    ⎜               ⎟   ⋅ ⎜ ⎟
#    ⎝q ⋅ p         s⎠     ⎝B⎠
def delta(vA: Vector2, vr: Vector2, vp: Vector2) -> Vector2:
    """for -vA1

    The `delta` function calculates the delta value using the given vectors `vA`, `vr`, and `vp`.

    :param vA: The parameter `vA` represents a 2D vector
    :type vA: Vector2
    :param vr: The parameter `vr` represents a 2D vector with components `r` and `q`
    :type vr: Vector2
    :param vp: vp is a Vector2 representing the vector p
    :type vp: Vector2
    :return: The function `delta` returns a `Vector2` object.

    Examples:
        >>> d = delta(Vector2(1, 2), Vector2(-2, 0), Vector2(4, 5))
        >>> print(d)
        <0.2, 0.4>
    """
    r, q = vr.x, vr.y
    p, s = vp.x, vp.y
    mp = Matrix2(Vector2(s, -p), Vector2(-p * q, p * r + s))
    return mp.mdot(vA) / mp.det()  # 6 mul's + 2 div's


def suppress_old(vA: Vector2, vA1: Vector2, vri: Vector2, vrj: Vector2):
    """Zero suppresion (original)

    The `suppress` function performs zero suppression on a given set of vectors using the Bairsow's
    method.

    :param vA: The parameter `vA` represents a 2D vector. It is not clear what this vector represents without further context
    :type vA: Vector2
    :param vA1: The parameter `vA1` represents a 2D vector
    :type vA1: Vector2
    :param vri: The parameter `vri` represents a vector with components `x` and `y`
    :type vri: Vector2
    :param vrj: The parameter `vrj` represents a vector `vrj` in the function `suppress()`. It is a `Vector2` object that represents the vector `vrj` in a mathematical context
    :type vrj: Vector2
    :return: The function `suppress` returns two values: `va` and `va1`.

    Reference:
        D. C. Handscomb, Computation of the latent roots of a Hessenberg matrix
        by Bairsow's method, Computer Journal, 5 (1962), pp. 139-141.

    Examples:
        >>> vA = Vector2(3, 3)
        >>> vA1 = Vector2(1, 2)
        >>> vri = Vector2(-2, 0)
        >>> vrj = Vector2(4, 5)
        >>> suppress_old(vA, vA1, vri, vrj)
        >>> dr = delta(vA, vri, vA1)
        >>> print(dr)
        <-16.78082191780822, 1.4383561643835616>
    """
    A, B = vA.x, vA.y
    A1, B1 = vA1.x, vA1.y
    vp = vri - vrj
    r, q = vri.x, vri.y
    p, s = vp.x, vp.y
    f = r * p + s
    qp = q * p
    e = f * s - qp * p
    a = A * s - B * p
    b = B * f - A * qp
    c = A1 * e - a
    d = B1 * e - b - a * p
    vA._x = a * e
    vA._y = b * e
    vA1._x = c * s - d * p
    vA1._y = d * f - c * qp
    # return delta(vA, vri, Vector2(vA1._x, -vA1._y))


def suppress(vA: Vector2, vA1: Vector2, vri: Vector2, vrj: Vector2):
    """Zero suppresion

    The `suppress` function performs zero suppression on a given set of vectors using the Bairsow's
    method.

    :param vA: The parameter `vA` represents a 2D vector. It is not clear what this vector represents without further context
    :type vA: Vector2
    :param vA1: The parameter `vA1` represents a 2D vector
    :type vA1: Vector2
    :param vri: The parameter `vri` represents a vector with components `x` and `y`
    :type vri: Vector2
    :param vrj: The parameter `vrj` represents a vector `vrj` in the function `suppress()`. It is a `Vector2` object that represents the vector `vrj` in a mathematical context
    :type vrj: Vector2
    :return: The function `suppress` returns two values: `va` and `va1`.

    Reference:
        D. C. Handscomb, Computation of the latent roots of a Hessenberg matrix
        by Bairsow's method, Computer Journal, 5 (1962), pp. 139-141.

    Examples:
        >>> vA = Vector2(3, 3)
        >>> vA1 = Vector2(1, 2)
        >>> vri = Vector2(-2, 0)
        >>> vrj = Vector2(4, 5)
        >>> vA, vA1 = suppress(vA, vA1, vri, vrj)
        >>> dr = delta(vA, vri, vA1)
        >>> print(dr)
        <-16.78082191780822, 1.4383561643835616>
    """
    vp = vri - vrj
    r, q = vri.x, vri.y
    p, s = vp.x, vp.y
    m_adjoint = Matrix2(Vector2(s, -p), Vector2(-p * q, p * r + s))
    e = m_adjoint.det()
    va = m_adjoint.mdot(vA)
    vc = vA1 * e - va
    vc._y -= va._x * p
    va *= e
    va1 = m_adjoint.mdot(vc)
    return va, va1


def suppress2(vA: Vector2, vA1: Vector2, vri: Vector2, vrj: Vector2):
    """Zero suppresion

    The `suppress` function performs zero suppression on a given set of vectors using the Bairsow's
    method.

    :param vA: The parameter `vA` represents a 2D vector. It is not clear what this vector represents without further context
    :type vA: Vector2
    :param vA1: The parameter `vA1` represents a 2D vector
    :type vA1: Vector2
    :param vri: The parameter `vri` represents a vector with components `x` and `y`
    :type vri: Vector2
    :param vrj: The parameter `vrj` represents a vector `vrj` in the function `suppress()`. It is a `Vector2` object that represents the vector `vrj` in a mathematical context
    :type vrj: Vector2
    :return: The function `suppress` returns two values: `va` and `va1`.

    Reference:
        D. C. Handscomb, Computation of the latent roots of a Hessenberg matrix
        by Bairsow's method, Computer Journal, 5 (1962), pp. 139-141.

    Examples:
        >>> vA = Vector2(3, 3)
        >>> vA1 = Vector2(1, 2)
        >>> vri = Vector2(-2, 0)
        >>> vrj = Vector2(4, 5)
        >>> vA, vA1 = suppress2(vA, vA1, vri, vrj)
        >>> dr = delta(vA, vri, vA1)
        >>> print(dr)
        <-16.78082191780822, 1.4383561643835616>
    """
    vp = vri - vrj
    r, q = vrj.x, vrj.y
    p, s = vp.x, vp.y
    m_adjoint = Matrix2(Vector2(s, -p), Vector2(-p * q, p * r + s))
    e = m_adjoint.det()
    # m_inv = m_adjoint / e
    va = vA * e
    va1 = vA1 * e - m_adjoint.mdot(vA)
    return va, va1


def horner_eval_f(coeffs: List, zval):
    """Polynomial evaluation using Horner's scheme

    The `horner_eval_f` function evaluates a polynomial using Horner's scheme.

    :param coeffs: A list of coefficients of a polynomial.
    :type coeffs: List
    :param zval: The `zval` parameter represents the value at which the polynomial is to be evaluated. It can be a float or a complex number
    :return: the value of the polynomial evaluated at the given value `zval`.

    Examples:
        >>> coeffs = [1, -8, -72, 382, 727, -2310]
        >>> horner_eval_f(coeffs, 3)
        960
    """
    return reduce(lambda res, coeff: res * zval + coeff, coeffs)


#                     n         n - 1
#        P(z) = c  ⋅ z  + c  ⋅ z      + ... + c
#                0         1                   n
#
#        P(z) = P (z) ⋅ ⎛z - z   ⎞ + A
#                1      ⎝     val⎠
def horner_eval(coeffs: List, zval) -> Tuple[Any, List]:
    """Polynomial evaluation using Horner's scheme

    The `horner_eval` function evaluates a polynomial using Horner's scheme and updates the coefficients
    list in place.

    :param coeffs: A list of coefficients of a polynomial.
    :type coeffs: List
    :param zval: The `zval` parameter represents the value at which the polynomial is to be evaluated. It can be a float or a complex number
    :return: the value of the polynomial evaluated at the given value `zval`.

    Examples:
        >>> coeffs = [1, -8, -72, 382, 727, -2310]
        >>> horner_eval(coeffs, 3)
        (960, [1, -5, -87, 121, 1090, 960])
        >>> horner_eval(coeffs, 3+0j)
        ((960+0j), [1, (-5+0j), (-87+0j), (121+0j), (1090+0j), (960+0j)])
    """
    coeffs = list(accumulate(coeffs, lambda res, coeff: res * zval + coeff))
    return coeffs[-1], coeffs
    # for i in range(degree):
    #    coeffs[i + 1] += coeffs[i] * zval


def horner_backward(coeffs: List, degree: int, val):
    """Polynomial evaluation using Horner's scheme

    The `horner_backward` function evaluates a polynomial using Horner's scheme and updates the coefficients
    list in place.

    :param coeffs: A list of coefficients of a polynomial.
    :type coeffs: List
    :param degree: The degree parameter represents the degree of the polynomial. It is an integer value that indicates the highest power of the variable in the polynomial
    :type degree: int
    :param zval: The `zval` parameter represents the value at which the polynomial is to be evaluated. It can be a float or a complex number
    :return: the value of the polynomial evaluated at the given value `zval`.

    Examples:
        >>> coeffs = [1.0, -6.7980, 2.9948, -0.043686, 0.000089248]
        >>> degree = len(coeffs) - 1
        >>> alpha = 6.3256
        >>> p_eval = horner_backward(coeffs, 4, alpha)
        >>> -p_eval * pow(alpha, 5)
        -0.013355264987140483
        >>> coeffs[3]
        0.006920331351966613
    """
    for i in range(2, degree + 2):
        coeffs[-i] -= coeffs[-(i - 1)]
        coeffs[-i] /= -val
    return coeffs[-(degree + 1)]


#
#                       ⎛ 2            ⎞
#        P(x) = P (x) ⋅ ⎝x  - r ⋅ x - q⎠ + A ⋅ x + B
#                1
#
#    Note: P(x) becomes the quotient after calling this function
def horner(coeffs: List[float], degree: int, vr: Vector2) -> Vector2:
    """Polynomial evaluation using Horner's scheme

    The `horner` function evaluates a polynomial using Horner's scheme and returns the result as a
    `Vector2` object.

    :param coeffs: The `coeffs` parameter is a list of coefficients of a polynomial. Each element in the list represents the coefficient of a term in the polynomial, starting from the highest degree term and going down to the constant term
    :type coeffs: List[float]
    :param degree: The degree parameter represents the degree of the polynomial. It determines the number of coefficients in the coeffs list
    :type degree: int
    :param vr: vr is a Vector2 object that represents the values of x and y in the polynomial expression
    :type vr: Vector2
    :return: The function `horner` returns a `Vector2` object.

    Examples:
        >>> coeffs = [1, -8, -72, 382, 727, -2310]
        >>> vp = horner(coeffs, 5, Vector2(-1, 6))  # x^2 + x - 6
        >>> coeffs
        [1, -9, -57, 385, 0, 0]
        >>> coeffs = [1, -8, -72, 382, 727, -2310]
        >>> vp = horner(coeffs, 5, Vector2(2, 3))  # x^2 - 2x - 3
        >>> coeffs
        [1, -6, -81, 202, 888, -1704]
    """
    for i in range(0, degree - 1):
        coeffs[i + 1] += coeffs[i] * vr.x
        coeffs[i + 2] += coeffs[i] * vr.y
    return Vector2(coeffs[degree - 1], coeffs[degree])


def initial_guess_orig(coeffs: List[float]) -> List[Vector2]:
    """Initial guess

    The `initial_guess` function calculates an initial guess for the roots of a polynomial equation using a specific algorithm.

    :param coeffs: The `coeffs` parameter is a list of floating-point numbers representing the coefficients of a polynomial.
                   The polynomial is of the form `coeffs[0] * x^n + coeffs[1] * x^(n-1) + ... + coeffs[n-1] * x + coeffs[n]`
    :type coeffs: List[float]
    :return: The function `initial_guess` returns a list of `Vector2` objects.

    Examples:
        >>> h = [10.0, 34.0, 75.0, 94.0, 150.0, 94.0, 75.0, 34.0, 10.0]
        >>> vr0s = initial_guess_orig(h)
    """
    degree = len(coeffs) - 1
    center = -coeffs[1] / (degree * coeffs[0])
    # p_eval = np.poly1d(coeffs)
    p_center = horner_eval_f(coeffs, center)
    radius = pow(abs(p_center), 1 / degree)
    m = center * center + radius * radius
    degree //= 2
    degree *= 2  # make even
    k = PI / degree
    temp = iter(radius * cos(k * i) for i in range(1, degree, 2))
    return [Vector2(2 * (center + t), -(m + 2 * center * t)) for t in temp]
    # vr0s = []
    # for i in range(1, degree, 2):
    #     temp = radius * cos(k * i)
    #     r0 = 2 * (center + temp)
    #     t0 = m + 2 * center * temp  # ???
    #     vr0s += [Vector2(r0, -t0)]
    # return vr0s


def initial_guess(coeffs: List[float]) -> List[Vector2]:
    """Initial guess

    The `initial_guess` function calculates an initial guess for the roots of a polynomial equation using a specific algorithm.

    :param coeffs: The `coeffs` parameter is a list of floating-point numbers representing the coefficients of a polynomial.
                   The polynomial is of the form `coeffs[0] * x^n + coeffs[1] * x^(n-1) + ... + coeffs[n-1] * x + coeffs[n]`
    :type coeffs: List[float]
    :return: The function `initial_guess` returns a list of `Vector2` objects.

    Examples:
        >>> h = [10.0, 34.0, 75.0, 94.0, 150.0, 94.0, 75.0, 34.0, 10.0]
        >>> vr0s = initial_guess(h)
    """
    degree = len(coeffs) - 1
    center = -coeffs[1] / (degree * coeffs[0])
    # p_eval = np.poly1d(coeffs)
    p_center = horner_eval_f(coeffs, center)
    radius = pow(abs(p_center), 1 / degree)
    m = center * center + radius * radius
    degree //= 2
    degree *= 2  # make even
    # k = PI / degree
    vgen = VdCorput(2)
    vgen.reseed(1)
    temp = iter(radius * cos(PI * vgen.pop()) for _ in range(1, degree, 2))
    return [Vector2(2 * (center + t), -(m + 2 * center * t)) for t in temp]
    # vr0s = []
    # for _ in range(1, degree, 2):
    #     temp = radius * cos(PI * vgen.pop())
    #     r0 = 2 * (center + temp)
    #     t0 = m + 2 * center * temp  # ???
    #     vr0s += [Vector2(r0, -t0)]
    # return vr0s


#            new                               -1
#        ⎛r ⎞      ⎛r ⎞   ⎛A'  ⋅ r  + B'    A' ⎞
#        ⎜ i⎟      ⎜ i⎟   ⎜  1    i     1     1⎟     ⎛A⎞
#        ⎜  ⎟    = ⎜  ⎟ - ⎜                    ⎟   ⋅ ⎜ ⎟
#        ⎜q ⎟      ⎜q ⎟   ⎜ A'  ⋅ q         B' ⎟     ⎝B⎠
#        ⎝ i⎠      ⎝ i⎠   ⎝  1    i           1⎠
#
#    where
#                         m
#                       _____
#                       ╲                         -1
#        ⎛A' ⎞   ⎛A ⎞    ╲    ⎛p  ⋅ r  + s     p  ⎞
#        ⎜  1⎟   ⎜ 1⎟     ╲   ⎜ ij   i    ij    ij⎟     ⎛A⎞
#        ⎜   ⎟ = ⎜  ⎟ -   ╱   ⎜                   ⎟   ⋅ ⎜ ⎟
#        ⎜B' ⎟   ⎜B ⎟    ╱    ⎜p  ⋅ q          s  ⎟     ⎝B⎠
#        ⎝  1⎠   ⎝ 1⎠   ╱     ⎝ ij   i          ij⎠
#                       ‾‾‾‾‾
#                       j ≠ i
#
#        ⎛p  ⎞   ⎛r ⎞   ⎛r ⎞
#        ⎜ ij⎟   ⎜ i⎟   ⎜ j⎟
#        ⎜   ⎟ = ⎜  ⎟ - ⎜  ⎟
#        ⎜s  ⎟   ⎜q ⎟   ⎜q ⎟
#        ⎝ ij⎠   ⎝ i⎠   ⎝ j⎠
def pbairstow_even(
    coeffs: List[float], vrs: List[Vector2], options=Options()
) -> Tuple[List[Vector2], int, bool]:
    """Parallel Bairstow's method

    The `pbairstow_even` function implements a parallel version of Bairstow's method for finding the
    roots of a polynomial.

    :param coeffs: The parameter `coeffs` is a list of floats representing the coefficients of a
                   polynomial. It represents the polynomial whose roots we want to find using Bairstow's method
    :type coeffs: List[float]
    :param vrs: The `vrs` parameter is a list of `Vector2` objects. Each `Vector2` object represents
                a complex number and is used as an initial guess for the roots of the polynomial equation. The
                length of the `vrs` list determines the number of roots to be found
    :type vrs: List[Vector2]
    :param options: The `options` parameter is an instance of the `Options` class, which is used to
                    specify various options for the Bairstow's method algorithm.
    :return: The function `pbairstow_even` returns a tuple containing the following elements:

    Examples:
        >>> h = [10.0, 34.0, 75.0, 94.0, 150.0, 94.0, 75.0, 34.0, 10.0]
        >>> vr0s = initial_guess(h)
        >>> vrs, niter, found = pbairstow_even(h, vr0s)
        >>> print(found)
        True
    """
    M = len(vrs)
    degree = len(coeffs) - 1
    converged = [False] * M
    robin = Robin(M)
    for niter in range(options.max_iters):
        tolerance = 0.0
        for i, (vri, ci) in enumerate(zip(vrs, converged)):
            if ci:
                continue
            coeffs1 = coeffs.copy()
            vA = horner(coeffs1, degree, vri)
            tol_i = max(abs(vA.x), abs(vA.y))
            if tol_i < options.tol_ind:
                converged[i] = True
                continue
            vA1 = horner(coeffs1, degree - 2, vri)
            tolerance = max(tol_i, tolerance)
            for j in robin.exclude(i):
                vA, vA1 = suppress(vA, vA1, vri, vrs[j])
            vrs[i] -= delta(vA, vri, vA1)
        if tolerance < options.tolerance:
            return vrs, niter, True
    return vrs, options.max_iters, False


def find_rootq(vr: Vector2) -> Tuple[Num, Num]:
    """find_rootq

    The function `find_rootq` solves a quadratic equation of the form x^2 - r*x - q = 0 and returns the
    two roots as a tuple.

        (x - x1)(x - x2) = x^2 - (x1 + x2) x + x1 * x2

    :param vr: The parameter `vr` is a Vector2 object that represents the coefficients of a quadratic
               equation. The `x` component of `vr` represents the coefficient of the linear term (`r`), and
               the `y` component represents the constant term (`q`) in the equation `x^2 - r*x - q = 0.`
    :type vr: Vector2
    :return: The function `find_rootq` returns a tuple containing the two roots of the quadratic
             equation x^2 - r*x - q = 0. The roots can be either floats or complex numbers, depending on the
             values of the input parameters.

    Examples:
        >>> vr = find_rootq(Vector2(5, -6))
        >>> print(vr)
        (3.0, 2.0)
    """
    # r, q = vr.x, vr.y
    hr = vr.x / 2
    d = hr * hr + vr.y
    if d < 0:
        x1 = hr + sqrt(-d) * 1j
    else:
        x1 = hr + (sqrt(d) if hr >= 0 else -sqrt(d))
    x2 = -vr.y / x1
    return x1, x2
