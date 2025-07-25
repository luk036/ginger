"""
rootfinding.py

This code is a collection of functions and classes designed to find the roots of polynomial equations. In mathematics, finding the roots of a polynomial means determining the values of x that make the polynomial equal to zero. This is a common problem in various fields like engineering, physics, and computer science.

The main purpose of this code is to implement the Bairstow's method, which is an algorithm for finding complex roots of polynomials. It takes as input the coefficients of a polynomial and initial guesses for the roots, and outputs the calculated roots of the polynomial.

The code starts by importing necessary modules and defining some utility classes and functions. The main algorithm is implemented in the pbairstow_even function. This function takes three inputs:

1. A list of coefficients representing the polynomial
2. A list of initial guesses for the roots
3. An optional Options object to control the algorithm's behavior

The output of pbairstow_even is a tuple containing:

1. A list of the calculated roots
2. The number of iterations performed
3. A boolean indicating whether the algorithm successfully converged

The algorithm works by iteratively refining the initial guesses for the roots. It uses a technique called "suppression" to improve the accuracy of each root estimate. The process continues until either the desired accuracy is achieved or the maximum number of iterations is reached.

Some important parts of the code include:

- The initial_guess function, which generates starting points for the algorithm
- The suppress function, which helps improve the accuracy of root estimates
- The horner function, which efficiently evaluates polynomials
- The delta function, which calculates adjustments to the root estimates

The code also includes helper functions for polynomial evaluation and manipulation, such as horner_eval and horner_backward.

Overall, this code provides a sophisticated tool for solving polynomial equations, even when the roots are complex numbers. It's designed to be efficient and accurate, making it useful for applications that require finding roots of high-degree polynomials.
"""

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
    """
    Configuration options for root-finding algorithms.

    This class provides control parameters for iterative root-finding methods:
    - max_iters: Maximum number of iterations allowed (default: 2000)
    - tolerance: Convergence tolerance for the algorithm (default: 1e-12)
    - tol_ind: Individual root tolerance threshold (default: 1e-15)

    These parameters allow fine-tuning of the algorithm's behavior and stopping criteria.
    """

    max_iters: int = 2000
    tolerance: float = 1e-12
    tol_ind: float = 1e-15
    # tol_suppress: float = 1e-1


#                      -1
#    ⎛r ⋅ p + s     p⎞     ⎛A⎞
#    ⎜               ⎟   ⋅ ⎜ ⎟
#    ⎝q ⋅ p         s⎠     ⎝B⎠
def delta(vA: Vector2, vr: Vector2, vp: Vector2) -> Vector2:
    """Calculate adjustment vector for Bairstow's method.

    The `delta` function computes the correction vector used in Bairstow's method to update
    root estimates. It solves a 2x2 linear system derived from polynomial division to find
    the optimal adjustment to current root estimates.

    :param vA: Residual vector (A,B) from polynomial division
    :type vA: Vector2
    :param vr: Current root estimate vector (r,q)
    :type vr: Vector2
    :param vp: Vector used in suppression calculations (p,s)
    :type vp: Vector2
    :return: Correction vector to adjust root estimates
    :rtype: Vector2

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
    """Original implementation of zero suppression in Bairstow's method.

    This function modifies the residual vectors vA and vA1 to suppress the influence
    of other roots (vrj) when estimating the current root (vri). This helps prevent
    interference between root estimates during iteration.

    Note: This is the original implementation that modifies vectors in-place.
    The newer version returns modified vectors instead.

    :param vA: Current residual vector (A,B)
    :type vA: Vector2
    :param vA1: First derivative residual vector (A1,B1)
    :type vA1: Vector2
    :param vri: Current root estimate being refined (ri,qi)
    :type vri: Vector2
    :param vrj: Another root estimate that might interfere (rj,qj)
    :type vrj: Vector2

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
    """Improved zero suppression for Bairstow's method.

    This function calculates modified residual vectors that account for interference
    from other roots in the system. It uses matrix operations to efficiently compute
    the suppression terms, providing better numerical stability than the original version.

    :param vA: Current residual vector (A,B)
    :type vA: Vector2
    :param vA1: First derivative residual vector (A1,B1)
    :type vA1: Vector2
    :param vri: Current root estimate being refined (ri,qi)
    :type vri: Vector2
    :param vrj: Another root estimate that might interfere (rj,qj)
    :type vrj: Vector2
    :return: Tuple of modified residual vectors (vA, vA1)
    :rtype: Tuple[Vector2, Vector2]

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
    """Alternative zero suppression implementation for Bairstow's method.

    This version uses a different approach to compute the suppression terms,
    potentially offering better performance or numerical properties in some cases.

    :param vA: Current residual vector (A,B)
    :type vA: Vector2
    :param vA1: First derivative residual vector (A1,B1)
    :type vA1: Vector2
    :param vri: Current root estimate being refined (ri,qi)
    :type vri: Vector2
    :param vrj: Another root estimate that might interfere (rj,qj)
    :type vrj: Vector2
    :return: Tuple of modified residual vectors (vA, vA1)
    :rtype: Tuple[Vector2, Vector2]

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
    """Evaluate polynomial using Horner's method (functional version).

    This function computes the value of a polynomial at a given point using
    Horner's method, which is more efficient than direct evaluation. It uses
    Python's reduce function for a concise implementation.

    :param coeffs: List of polynomial coefficients in descending order of degree
    :type coeffs: List
    :param zval: Point at which to evaluate the polynomial (can be complex)
    :return: Value of the polynomial at zval
    :rtype: Same type as zval (float or complex)

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
    """Evaluate polynomial and return intermediate coefficients.

    This function uses Horner's method to evaluate a polynomial and also returns
    the intermediate coefficients that result from the synthetic division process.
    These coefficients can be used for further computations like derivatives.

    :param coeffs: List of polynomial coefficients in descending order of degree
    :type coeffs: List
    :param zval: Point at which to evaluate the polynomial (can be complex)
    :return: Tuple containing:
             - Value of polynomial at zval
             - List of intermediate coefficients from synthetic division
    :rtype: Tuple[Any, List]

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


# def horner_backward(coeffs: List, degree: int, val):
#     """Polynomial evaluation using Horner's scheme

#     The `horner_backward` function evaluates a polynomial using Horner's scheme and updates the coefficients
#     list in place.

#     :param coeffs: A list of coefficients of a polynomial.
#     :type coeffs: List
#     :param degree: The degree parameter represents the degree of the polynomial. It is an integer value that indicates the highest power of the variable in the polynomial
#     :type degree: int
#     :param zval: The `zval` parameter represents the value at which the polynomial is to be evaluated. It can be a float or a complex number
#     :return: the value of the polynomial evaluated at the given value `zval`.

#     Examples:
#         >>> coeffs = [1.0, -6.7980, 2.9948, -0.043686, 0.000089248]
#         >>> degree = len(coeffs) - 1
#         >>> alpha = 6.3256
#         >>> p_eval = horner_backward(coeffs, 4, alpha)
#         >>> -p_eval * pow(alpha, 5)
#         -0.013355264987140483
#         >>> coeffs[3]
#         0.006920331351966613
#     """
#     for i in range(2, degree + 2):
#         coeffs[-i] -= coeffs[-(i - 1)]
#         coeffs[-i] /= -val
#     return coeffs[-(degree + 1)]


#
#                       ⎛ 2            ⎞
#        P(x) = P (x) ⋅ ⎝x  - r ⋅ x - q⎠ + A ⋅ x + B
#                1
#
#    Note: P(x) becomes the quotient after calling this function
def horner(coeffs: List[float], degree: int, vr: Vector2) -> Vector2:
    """Evaluate quadratic polynomial factor and return remainder.

    This specialized version of Horner's method evaluates a polynomial divided by
    a quadratic factor (x² - r·x - q). It returns the linear remainder (A·x + B)
    and modifies the coefficients array to contain the quotient polynomial.

    :param coeffs: List of polynomial coefficients in descending order
    :type coeffs: List[float]
    :param degree: Degree of the polynomial (must be ≥ 2)
    :type degree: int
    :param vr: Vector representing quadratic factor coefficients (r,q)
    :type vr: Vector2
    :return: Remainder vector (A,B)
    :rtype: Vector2

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


# def initial_guess_orig(coeffs: List[float]) -> List[Vector2]:
#     """Initial guess

#     The `initial_guess` function calculates an initial guess for the roots of a polynomial equation using a specific algorithm.

#     :param coeffs: The `coeffs` parameter is a list of floating-point numbers representing the coefficients of a polynomial.
#                    The polynomial is of the form `coeffs[0] * x^n + coeffs[1] * x^(n-1) + ... + coeffs[n-1] * x + coeffs[n]`
#     :type coeffs: List[float]
#     :return: The function `initial_guess` returns a list of `Vector2` objects.

#     Examples:
#         >>> h = [10.0, 34.0, 75.0, 94.0, 150.0, 94.0, 75.0, 34.0, 10.0]
#         >>> vr0s = initial_guess_orig(h)
#     """
#     degree = len(coeffs) - 1
#     center = -coeffs[1] / (degree * coeffs[0])
#     # p_eval = np.poly1d(coeffs)
#     poly_c = horner_eval_f(coeffs, center)
#     radius = pow(abs(poly_c), 1 / degree)
#     m = center * center + radius * radius
#     degree //= 2
#     degree *= 2  # make even
#     k = PI / degree
#     temp = iter(radius * cos(k * i) for i in range(1, degree, 2))
#     return [Vector2(2 * (center + t), -(m + 2 * center * t)) for t in temp]


def initial_guess(coeffs: List[float]) -> List[Vector2]:
    """Generate initial root estimates for Bairstow's method.

    This function creates reasonable starting points for the root-finding algorithm
    by distributing estimates around a circle in the complex plane. The circle's
    center and radius are determined from the polynomial's coefficients.

    :param coeffs: List of polynomial coefficients in descending order
    :type coeffs: List[float]
    :return: List of initial root estimates as Vector2 objects (r,q pairs)
    :rtype: List[Vector2]

    Examples:
        >>> h = [10.0, 34.0, 75.0, 94.0, 150.0, 94.0, 75.0, 34.0, 10.0]
        >>> vr0s = initial_guess(h)
    """
    degree: int = len(coeffs) - 1
    center: float = -coeffs[1] / (degree * coeffs[0])
    poly_c: float = horner_eval_f(coeffs, center)
    radius: float = pow(abs(poly_c), 1.0 / degree)
    m: float = center * center + radius * radius
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
    """Parallel implementation of Bairstow's root-finding method.

    This function implements a parallel version of Bairstow's method for finding
    all roots of a polynomial simultaneously. It works by iteratively improving
    estimates of quadratic factors of the polynomial.

    :param coeffs: List of polynomial coefficients in descending order
    :type coeffs: List[float]
    :param vrs: Initial estimates for quadratic factors (as Vector2 pairs)
    :type vrs: List[Vector2]
    :param options: Configuration parameters for the algorithm
    :type options: Options
    :return: Tuple containing:
             - Final root estimates
             - Number of iterations performed
             - Convergence status (True if converged)
    :rtype: Tuple[List[Vector2], int, bool]

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
    """Solve quadratic equation x² - r·x - q = 0.

    This function finds the roots of a quadratic equation represented by
    the Vector2 vr, where vr.x is r and vr.y is q in the equation above.
    It handles both real and complex roots appropriately.

    :param vr: Vector containing quadratic coefficients (r,q)
    :type vr: Vector2
    :return: Tuple containing the two roots (real or complex)
    :rtype: Tuple[Num, Num]

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
