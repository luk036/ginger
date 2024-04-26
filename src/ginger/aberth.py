from cmath import exp
from math import pi
from typing import List, Tuple, Union

from lds_gen.lds import VdCorput

from .robin import Robin

# from pytest import approx
from .rootfinding import Options, horner_eval, horner_eval_f

FoC = Union[float, complex]
PI = pi


def horner_backward(coeffs1: List, degree: int, alpha: FoC) -> FoC:
    """
    The `horner_backward` function evaluates a polynomial using the Horner's method in backward form.

    :param coeffs1: The parameter `coeffs1` is a list of coefficients of a polynomial in descending order of degree. For example, if the polynomial is `3x^3 - 2x^2 + 5x - 1`, then `coeffs1` would be `[3, -2, 5, -1]`
    :type coeffs1: List
    :param degree: The degree of the polynomial, which is the highest power of the variable in the polynomial. For example, if the polynomial is 3x^2 + 2x + 1, then the degree is 2
    :type degree: int
    :param alpha: The value of alpha is a constant that is used in the Horner's method for backward polynomial evaluation. It is typically a scalar value
    :type alpha: FoC
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


def initial_aberth(coeffs: List[FoC]) -> List[complex]:
    """
    The `initial_aberth` function calculates the initial guesses for the roots of a polynomial using the
    Aberth method.

    :param coeffs: The `coeffs` parameter is a list of coefficients of a polynomial. Each
                   element in the list represents the coefficient of a term in the polynomial, starting
                   from the highest degree term down to the constant term. For example, if the polynomial is
                   `3x^3 - 2x^2 + 5x - 1`, then `coeffs` would be `[3, -2, 5, -1]`
    :type coeffs: List[FoC]
    :return: The function `initial_aberth` returns a list of complex numbers.

    Examples:
        >>> h = [5.0, 2.0, 9.0, 6.0, 2.0]
        >>> z0s = initial_aberth(h)
    """
    degree: int = len(coeffs) - 1
    center: FoC = -coeffs[1] / (degree * coeffs[0])
    p_center: FoC = horner_eval_f(coeffs, center)
    re: FoC = pow(-p_center, 1.0 / degree)
    k = 2 * PI / degree
    return [center + re * exp(k * (0.25 + i) * 1j) for i in range(degree)]
    # z0s: List[complex] = []
    # for i in range(degree):
    #     theta = k * (0.25 + i)
    #     z0s += [center + re * exp(theta * 1j)]
    # return z0s


def initial_aberth_orig(coeffs: List[FoC]) -> List[complex]:
    """
    The function `initial_aberth_orig` calculates the initial approximations for the roots of a
    polynomial using the Aberth method.

    :param coeffs: The `coeffs` parameter is a list of coefficients of a polynomial. Each
                   element in the list represents the coefficient of a term in the polynomial, starting
                   from the highest degree term down to the constant term. For example, if the polynomial
                   is `3x^3 - 2x^2 + 5x - 1`, then `coeffs` would be `[3, -2, 5, -1]`
    :type coeffs: List[FoC]
    :return: The function `initial_aberth_orig` returns a list of complex numbers.

    Examples:
        >>> h = [5.0, 2.0, 9.0, 6.0, 2.0]
        >>> z0s = initial_aberth_orig(h)
    """
    degree: int = len(coeffs) - 1
    center: FoC = -coeffs[1] / (degree * coeffs[0])
    p_center: FoC = horner_eval_f(coeffs, center)
    re: FoC = pow(-p_center, 1.0 / degree)
    k = 2 * PI / degree
    return [center + re * exp(k * (0.25 + i) * 1j) for i in range(degree)]
    # z0s: List[complex] = []
    # for i in range(degree):
    #     theta = k * (0.25 + i)
    #     z0s += [center + re * exp(theta * 1j)]
    # return z0s


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
    coeffs: List[FoC], zs: List[complex], options: Options = Options()
) -> Tuple[List[complex], int, bool]:
    """
    The `aberth` function implements Aberth's method for polynomial root-finding.

    :param coeffs: The `coeffs` parameter is a list of coefficients of a polynomial. The
                   coefficients are ordered from highest degree to lowest degree. For example, if the
                   polynomial is `3x^2 + 2x + 1`, then the `coeffs` list would be `[3, 2, 1]`
    :type coeffs: List[FoC]
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
    M = len(zs)
    # degree = len(coeffs) - 1
    converged = [False] * M
    robin = Robin(M)
    for niter in range(options.max_iters):
        tolerance = 0.0
        for i, (zi, ci) in enumerate(zip(zs, converged)):
            if ci:
                continue
            p_eval, coeffs1 = horner_eval(coeffs, zi)
            tol_i = abs(p_eval)
            if tol_i < options.tol_ind:
                converged[i] = True
                continue
            p1_eval, _ = horner_eval(coeffs1[:-1], zi)
            tolerance = max(tol_i, tolerance)
            # for j in filter(lambda j: j != i, range(M)):  # exclude i
            for j in robin.exclude(i):
                p1_eval -= p_eval / (zi - zs[j])
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
    degree: int = len(coeffs) - 1
    re: float = pow(abs(coeffs[-1]), 1.0 / degree)
    if abs(re) > 1:
        re = 1 / re
    degree //= 2
    vgen = VdCorput(2)
    vgen.reseed(1)
    return [re * exp(2 * PI * vgen.pop() * 1j) for _ in range(degree)]
    # z0s = []
    # for _ in range(degree):
    #     vdc = 2 * PI * vgen.pop()
    #     z0s += [re * exp(vdc * 1j)]
    # return z0s


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
    re: float = pow(abs(coeffs[-1]), 1.0 / degree)
    # center = -coeffs[1] / (degree * coeffs[0])
    # p_center = horner_eval(coeffs.copy(), degree, center)
    # re = (-p_center) ** (1.0 / degree)
    if abs(re) > 1:
        re = 1 / re
    degree //= 2
    k = 2 * PI / degree
    return [re * exp(k * (0.25 + i) * 1j) for i in range(degree)]
    # z0s = []
    # for i in range(degree):
    #     theta = k * (0.25 + i)
    #     z0s += [re * exp(theta * 1j)]
    # return z0s


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
    M: int = len(zs)
    converged: List[bool] = [False] * M
    robin = Robin(M)
    for niter in range(options.max_iters):
        tolerance: float = 0.0
        for i, (zi, ci) in enumerate(zip(zs, converged)):
            if ci:
                continue
            p_eval, coeffs1 = horner_eval(coeffs, zi)
            tol_i = abs(p_eval)
            if tol_i < options.tol_ind:
                converged[i] = True
                continue
            p1_eval, _ = horner_eval(coeffs1[:-1], zi)
            tolerance = max(tol_i, tolerance)
            for j in robin.exclude(i):
                zj = zs[j]
                p1_eval -= p_eval / (zi - zj)
                zsn = 1.0 / zj
                p1_eval -= p_eval / (zi - zsn)
            zs[i] -= p_eval / p1_eval
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
