"""
Aberth's Method for Polynomial Root Finding

This code implements Aberth's method, which is an algorithm for finding the roots of polynomials. In simple terms, it's a way to solve equations like x^3 + 2x^2 - 5x + 3 = 0, finding the values of x that make the equation true.

The main input for this code is a list of coefficients that represent a polynomial. For example, [1, 2, -5, 3] would represent the polynomial x^3 + 2x^2 - 5x + 3. The code also takes initial guesses for where the roots might be.

The output is a list of complex numbers that represent the roots of the polynomial. These are the solutions to the equation. The code also returns the number of iterations it took to find the roots and whether it was successful in finding them within the specified tolerance.

To achieve its purpose, the code uses an iterative process. It starts with initial guesses for the roots and then repeatedly improves these guesses until they're close enough to the actual roots. The main algorithm, Aberth's method, is implemented in the aberth function. This function uses a clever mathematical formula to update each guess based on the current polynomial value and its derivative at that point, as well as the positions of all the other guesses.

The code includes several variations of the algorithm. There's a basic version (aberth), a multithreaded version for faster computation (aberth_mt), and versions that use autocorrelation (aberth_autocorr and aberth_autocorr_mt). These autocorrelation versions are designed to work better for certain types of polynomials.

An important part of the process is finding good initial guesses for the roots. The code includes several functions for this, like initial_aberth and initial_aberth_autocorr. These functions use mathematical insights about where roots are likely to be located to make educated guesses.

The code also includes helper functions like horner_eval and horner_backward which are efficient ways to evaluate polynomials and their derivatives.

Overall, this code provides a comprehensive toolkit for finding the roots of polynomials using Aberth's method, with various optimizations and variations to handle different scenarios efficiently.
"""

from concurrent.futures import ThreadPoolExecutor
from math import cos, sin, pi
from typing import List, Tuple

from lds_gen.lds import Circle
# from mywheel.robin import Robin

# from pytest import approx
from .rootfinding import Options, horner_eval, horner_eval_f

TWO_PI: float = 2.0 * pi


def horner_backward(coeffs1: List, degree: int, alpha: complex) -> complex:
    """
    The `horner_backward` function evaluates a polynomial using the Horner's method in backward form.

    :param coeffs1: The parameter `coeffs1` is a list of coefficients of a polynomial in descending order of degree. For example, if the polynomial is `3x^3 - 2x^2 + 5x - 1`, then `coeffs1` would be `[3, -2, 5, -1]`
    :type coeffs1: List
    :param degree: The degree of the polynomial, which is the highest power of the variable in the polynomial. For example, if the polynomial is 3x^2 + 2x + 1, then the degree is 2
    :type degree: int
    :param alpha: The value of alpha is a constant that is used in the Horner's method for backward polynomial evaluation. It is typically a scalar value
    :type alpha: complex
    :return: The function `horner_backward` returns the value of the polynomial evaluated at the given alpha value.

    Examples:
        >>> coeffs = [1.0, -6.7980, 2.9948, -0.043686, 0.000089248]
        >>> degree = len(coeffs) - 1
        >>> alpha = 6.3256
        >>> p_eval = horner_backward(coeffs, degree, alpha)
        >>> -p_eval * pow(alpha, 5)
        -0.013355264987140483
        >>> coeffs[3]
        0.006920331351966613
    """
    for i in range(2, degree + 2):
        coeffs1[-i] -= coeffs1[-(i - 1)]
        coeffs1[-i] /= -alpha
    return coeffs1[-(degree + 1)]


def initial_aberth(coeffs: List[float]) -> List[complex]:
    """
    The `initial_aberth` function calculates the initial guesses for the roots of a polynomial using the
    Aberth method.

    :param coeffs: The `coeffs` parameter is a list of coefficients of a polynomial. Each
                   element in the list represents the coefficient of a term in the polynomial, starting
                   from the highest degree term down to the constant term. For example, if the polynomial is
                   `3x^3 - 2x^2 + 5x - 1`, then `coeffs` would be `[3, -2, 5, -1]`
    :type coeffs: List[float]
    :return: The function `initial_aberth` returns a list of complex numbers.

    Examples:
        >>> h = [5.0, 2.0, 9.0, 6.0, 2.0]
        >>> z0s = initial_aberth(h)
    """
    degree: int = len(coeffs) - 1
    center: float = -coeffs[1] / (degree * coeffs[0])
    poly_c: float = horner_eval_f(coeffs, center)
    radius: float | complex = pow(-poly_c, 1.0 / degree)
    # radius: float = pow(abs(poly_c), 1.0 / degree)
    c_gen = Circle(2)
    return [
        center + radius * complex(x, y) for y, x in (c_gen.pop() for _ in range(degree))
    ]


def initial_aberth_orig(coeffs: List[float]) -> List[complex]:
    """
    The function `initial_aberth_orig` calculates the initial approximations for the roots of a
    polynomial using the Aberth method.

    :param coeffs: The `coeffs` parameter is a list of coefficients of a polynomial. Each
                   element in the list represents the coefficient of a term in the polynomial, starting
                   from the highest degree term down to the constant term. For example, if the polynomial
                   is `3x^3 - 2x^2 + 5x - 1`, then `coeffs` would be `[3, -2, 5, -1]`
    :type coeffs: List[float]
    :return: The function `initial_aberth_orig` returns a list of complex numbers.

    Examples:
        >>> h = [5.0, 2.0, 9.0, 6.0, 2.0]
        >>> z0s = initial_aberth_orig(h)
    """
    degree: int = len(coeffs) - 1
    center: float = -coeffs[1] / (degree * coeffs[0])
    poly_c: float = horner_eval_f(coeffs, center)
    radius: float | complex = pow(-poly_c, 1.0 / degree)
    k = TWO_PI / degree
    return [
        center + radius * (cos(theta) + sin(theta) * 1j)
        for theta in (k * (0.25 + i) for i in range(degree))
    ]


def aberth_job(
    coeffs: List[float],
    i: int,
    zsc: List[complex],
) -> Tuple[float, int, complex]:
    zi = zsc[i]
    p_eval, coeffs1 = horner_eval(coeffs, zi)
    tol_i = abs(p_eval)
    p1_eval, _ = horner_eval(coeffs1[:-1], zi)
    for j, zj in enumerate(zsc):
        if i != j:
            p1_eval -= p_eval / (zi - zj)
    zi -= p_eval / p1_eval
    return tol_i, i, zi


def aberth_mt(
    coeffs: List[float], zs: List[complex], options: Options = Options()
) -> Tuple[List[complex], int, bool]:
    with ThreadPoolExecutor() as executor:
        for niter in range(options.max_iters):
            tolerance = 0.0
            futures = []

            for i in range(len(zs)):
                futures.append(executor.submit(aberth_job, coeffs, i, zs))

            for future in futures:
                tol_i, i, zi = future.result()
                if tol_i > tolerance:
                    tolerance = tol_i
                zs[i] = zi

            if tolerance < options.tolerance:
                return zs, niter, True

    return zs, options.max_iters, False


#
#                     P ⎛z ⎞
#          new          ⎝ i⎠
#         z    = z  - ───────
#          i      i   P' ⎛z ⎞
#                        ⎝ i⎠
#     where
#                               n
#                             _____
#                             ╲
#                              ╲    P ⎛z ⎞
#                               ╲     ⎝ i⎠
#         P' ⎛z ⎞ = P  ⎛z ⎞ -   ╱   ───────
#            ⎝ i⎠    1 ⎝ i⎠    ╱    z  - z
#                             ╱      i    j
#                             ‾‾‾‾‾
#                             j ≠ i
def aberth(
    coeffs: List[float], zs: List[complex], options: Options = Options()
) -> Tuple[List[complex], int, bool]:
    """
    The `aberth` function implements Aberth's method for polynomial root-finding.

    :param coeffs: The `coeffs` parameter is a list of coefficients of a polynomial. The
                   coefficients are ordered from highest degree to lowest degree. For example, if the
                   polynomial is `3x^2 + 2x + 1`, then the `coeffs` list would be `[3, 2, 1]`
    :type coeffs: List[float]
    :param zs: The `zs` parameter in the `aberth` function represents the initial guesses for
               the roots of the polynomial. It is a list of complex numbers. Each complex number represents
               an initial guess for a root of the polynomial
    :type zs: List[complex]
    :param options: The `options` parameter is an instance of the `Options` class, which contains
                    various options for the Aberth's method algorithm. It is an optional parameter, and if not
                    provided, it will default to an instance of the `Options` class with default values
    :type options: Options
    :return: The function `aberth` returns a tuple containing three elements:
               1. `zs`: a list of complex numbers representing the approximate roots of the polynomial.
               2. `niter`: an integer representing the number of iterations performed by Aberth's method.
               3. `found`: a boolean value indicating whether the roots were found within the specified tolerance.

    Examples:
        >>> h = [5.0, 2.0, 9.0, 6.0, 2.0]
        >>> z0s = initial_aberth(h)
        >>> opt = Options()
        >>> opt.tolerance = 1e-8
        >>> zs, niter, found = aberth(h, z0s, opt)
        >>> found
        True
    """
    for niter in range(options.max_iters):
        tolerance = 0.0
        for i, zi in enumerate(zs):
            p_eval, coeffs1 = horner_eval(coeffs, zi)
            tol_i = abs(p_eval)
            p1_eval, _ = horner_eval(coeffs1[:-1], zi)
            tolerance = max(tol_i, tolerance)
            for j, zj in enumerate(zs):
                if i != j:
                    p1_eval -= p_eval / (zi - zj)
            zs[i] -= p_eval / p1_eval
        if tolerance < options.tolerance:
            return zs, niter, True
    return zs, options.max_iters, False


def initial_aberth_autocorr(coeffs: List[float]) -> List[complex]:
    """
    The function `initial_aberth_autocorr` calculates the initial values for the Aberth method for
    finding the roots of a polynomial.

    :param coeffs: The `coeffs` parameter is a list of coefficients of a polynomial. The coefficients
                   are ordered from highest degree to lowest degree. For example, if the polynomial
                   is `3x^2 + 2x + 1`, then the `coeffs` list would be `[3, 2, 1]`
    :type coeffs: List[float]
    :return: The function `initial_aberth_autocorr` returns a list of complex numbers.

    Examples:
        >>> h = [5.0, 2.0, 9.0, 6.0, 2.0]
        >>> z0s = initial_aberth_autocorr(h)
    """

    degree: int = len(coeffs) - 1  # assume even
    center: float = -coeffs[1] / (degree * coeffs[0])
    poly_c: float = horner_eval_f(coeffs, center)
    radius: float | complex = pow(-poly_c, 1.0 / degree)
    # radius: float | complex = pow(-coeffs[-1], 1.0 / degree)
    if abs(radius) > 1.0:
        radius = 1.0 / radius
    c_gen = Circle(2)
    return [
        center + radius * complex(x, y)
        for y, x in (c_gen.pop() for _ in range(degree // 2))
    ]


def initial_aberth_autocorr_orig(coeffs: List[float]) -> List[complex]:
    """
    The function `initial_aberth_autocorr_orig` calculates the initial guesses for the roots of a
    polynomial using the Aberth method.

    :param coeffs: The `coeffs` parameter is a list of coefficients of a polynomial. The
                   coefficients are ordered from highest degree to lowest degree. For example, if the polynomial
                   is `3x^2 + 2x + 1`, then the `coeffs` list would be `[3, 2, 1]`
    :type coeffs: List[float]
    :return: The function `initial_aberth_autocorr_orig` returns a list of complex numbers.

    Examples:
        >>> h = [5.0, 2.0, 9.0, 6.0, 2.0]
        >>> z0s = initial_aberth_autocorr_orig(h)
    """
    degree: int = len(coeffs) - 1
    center: float = -coeffs[1] / (degree * coeffs[0])
    poly_c: float = horner_eval_f(coeffs, center)
    radius: float = pow(abs(poly_c), 1.0 / degree)
    # radius: float = pow(abs(coeffs[-1]), 1.0 / degree)
    if abs(radius) > 1:
        radius = 1 / radius
    degree //= 2
    k = TWO_PI / degree
    return [
        center + radius * (cos(theta) + sin(theta) * 1j)
        for theta in (k * (0.25 + i) for i in range(degree))
    ]


def aberth_autocorr(
    coeffs: List[float], zs: List[complex], options=Options()
) -> Tuple[List[complex], int, bool]:
    """
    The `aberth_autocorr` function implements the Aberth method for finding the roots of a polynomial
    using autocorrelation.

    :param coeffs: The `coeffs` parameter is a list of coefficients of a polynomial. The coefficients
                   are ordered from highest degree to lowest degree. For example, if the polynomial
                   is `3x^2 + 2x + 1`, then the `coeffs` list would be `[3, 2, 1]`
    :type coeffs: List[float]
    :param zs: The `zs` parameter is a list of complex numbers. It represents the
               initial guesses for the roots of a polynomial
    :type zs: List[complex]
    :param options: The `options` parameter is an instance of the `Options` class, which contains
                    various options for the algorithm. It is an optional parameter and if not provided, it will
                    default to an instance of the `Options` class with default values
    :return: The function `aberth_autocorr` returns a tuple containing the following elements:

    Examples:
        >>> h = [5.0, 2.0, 9.0, 6.0, 2.0]
        >>> z0s = initial_aberth_autocorr(h)
        >>> zs, niter, found = aberth_autocorr(h, z0s)
        >>> opt = Options()
        >>> opt.tolerance = 1e-8
        >>> zs, niter, found = aberth_autocorr(h, z0s, opt)
    """
    for niter in range(options.max_iters):
        tolerance: float = 0.0
        for i, zi in enumerate(zs):
            p_eval, coeffs1 = horner_eval(coeffs, zi)
            tol_i = abs(p_eval)
            p1_eval, _ = horner_eval(coeffs1[:-1], zi)
            tolerance = max(tol_i, tolerance)
            for j, zj in enumerate(zs):
                if i == j:
                    continue
                p1_eval -= p_eval / (zi - zj)
                p1_eval -= p_eval / (zi - 1.0 / zj)
            zs[i] -= p_eval / p1_eval
        if tolerance < options.tolerance:
            return zs, niter, True
    return zs, options.max_iters, False


def aberth_autocorr_job(
    coeffs: List[float],
    i: int,
    zsc: List[complex],
) -> Tuple[float, int, complex]:
    zi = zsc[i]
    p_eval, coeffs1 = horner_eval(coeffs, zi)
    tol_i = abs(p_eval)
    p1_eval, _ = horner_eval(coeffs1[:-1], zi)
    for j, zj in enumerate(zsc):
        if i != j:
            p1_eval -= p_eval / (zi - zj)
            p1_eval -= p_eval / (zi - 1.0 / zj)
    zi -= p_eval / p1_eval
    return tol_i, i, zi


def aberth_autocorr_mt(
    coeffs: List[float], zs: List[complex], options: Options = Options()
) -> Tuple[List[complex], int, bool]:
    with ThreadPoolExecutor() as executor:
        for niter in range(options.max_iters):
            tolerance = 0.0
            futures = []

            for i in range(len(zs)):
                futures.append(executor.submit(aberth_autocorr_job, coeffs, i, zs))

            for future in futures:
                tol_i, i, zi = future.result()
                if tol_i > tolerance:
                    tolerance = tol_i
                zs[i] = zi

            if tolerance < options.tolerance:
                return zs, niter, True

    return zs, options.max_iters, False


# def test_aberth():
#     h = [5.0, 2.0, 9.0, 6.0, 2.0]
#     z0s = initial_aberth(h)
#     zs, niter, found = aberth(h, z0s)
#     assert (niter == 2)
#     assert (found)
#     zs, niter, found = aberth(h, z0s, Options(tolerance=1e-10))
#     assert (niter == 2)
#     assert (found)
#     zs, niter, found = aberth(h, z0s, Options(max_iters=1))
#     assert (niter == 1)
#     assert (found)
#     zs, niter, found = aberth(h, z0s, Options(max_iters=1, tolerance=1e-10))
#     assert (niter == 1)
#     assert (found)
#     zs, niter, found = aberth(h, z0s, Options(max_iters=1, tolerance=1e-11))
#     assert (niter == 0)
