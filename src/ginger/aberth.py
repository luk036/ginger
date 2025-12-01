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

import math
from concurrent.futures import ThreadPoolExecutor
from math import cos, sin
from typing import List, Sequence, Tuple, Union

from lds_gen.lds import Circle

from .rootfinding import Options, horner_eval, horner_eval_f

Num = Union[float, complex]
# from mywheel.robin import Robin


def horner_backward(coeffs1: List, degree: int, alpha: complex) -> complex:
    """
    Backward polynomial evaluation using Horner's method for root refinement.
    Evaluates polynomial at x=α using coefficients in reverse order.
    This implementation modifies coefficients in-place for efficiency.

    The `horner_backward` function evaluates a polynomial using the Horner's method in backward form.
    This is particularly useful for root refinement in iterative methods like Aberth's.
    It works by transforming the polynomial coefficients to center them around α,
    which helps in accurately evaluating the polynomial and its derivatives at α.

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


def initial_aberth(coeffs: Sequence[float]) -> List[complex]:
    """
    Generates initial root guesses using geometric distribution around a center point.
    Calculates center from polynomial coefficients and radius from evaluation at center.
    Uses low-discrepancy sequence (Circle generator) for even angular distribution.

    The `initial_aberth` function calculates the initial guesses for the roots of a polynomial using the
    Aberth method. It computes a center point based on the polynomial coefficients and then
    distributes initial guesses evenly around a circle centered at this point. The radius
    is determined by evaluating the polynomial at the center point and taking the nth root.

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
    poly_c: Num = horner_eval_f(coeffs, center)
    radius: float | complex = pow(-poly_c, 1.0 / degree)
    # radius: float = pow(abs(poly_c), 1.0 / degree)
    c_gen = Circle(2)
    return [
        center + radius * complex(x, y)
        for y, x in (c_gen.pop() for _ in range(degree))
        #    ^------ Note!
    ]


def initial_aberth_orig(coeffs: Sequence[float]) -> List[complex]:
    """
    Original implementation of initial guess generation using trigonometric distribution.
    Places roots equally spaced around a circle with calculated radius and center.
    Includes angular offset of 0.25 to avoid alignment with coordinate axes.

    The function `initial_aberth_orig` calculates the initial approximations for the roots of a
    polynomial using the Aberth method. This version uses trigonometric functions to distribute
    the initial guesses evenly around a circle, with a small angular offset to prevent roots
    from aligning with the coordinate axes.

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
    poly_c: Num = horner_eval_f(coeffs, center)
    radius: float | complex = pow(-poly_c, 1.0 / degree)
    k = 2.0 * math.pi / degree
    return [
        center + radius * (cos(theta) + sin(theta) * 1j)
        for theta in (k * (0.25 + i) for i in range(degree))
    ]


def aberth_mt(
    coeffs: Sequence[float], zs: List[complex], options: Options = Options()
) -> Tuple[List[complex], int, bool]:
    """
    Multithreaded implementation of Aberth's method.
    Uses ThreadPoolExecutor to parallelize root updates across available CPUs.
    Maintains convergence checking in main thread while parallelizing computations.

    This function implements Aberth's method for finding polynomial roots using multiple threads.
    Each root update is performed in parallel, which can significantly speed up computation
    for high-degree polynomials. The function maintains the same mathematical operations as
    the single-threaded version but distributes the workload across available processors.

    :param coeffs: List of polynomial coefficients in descending order of degree
    :param zs: Initial guesses for the roots (complex numbers)
    :param options: Configuration options including max iterations and tolerance
    :return: Tuple containing:
             - List of refined roots
             - Number of iterations performed
             - Boolean indicating whether convergence was achieved
    """

    def aberth_job(
        i: int,
    ) -> Tuple[float, int, complex]:
        zi = zs[i]
        p_eval, coeffs1 = horner_eval(coeffs, zi)
        tol_i = abs(p_eval)
        p1_eval, _ = horner_eval(coeffs1[:-1], zi)
        for j, zj in enumerate(zs):
            if i != j:
                p1_eval -= p_eval / (zi - zj)
        zi -= p_eval / p1_eval
        return tol_i, i, zi

    with ThreadPoolExecutor() as executor:
        for niter in range(options.max_iters):
            tolerance = 0.0
            futures = []

            for i in range(len(zs)):
                futures.append(executor.submit(aberth_job, i))

            for future in futures:
                tol_i, i, zi = future.result()
                if tol_i > tolerance:
                    tolerance = tol_i
                zs[i] = zi

            if tolerance < options.tolerance:
                return zs, niter, True

    return zs, options.max_iters, False


def aberth(
    coeffs: Sequence[float], zs: List[complex], options: Options = Options()
) -> Tuple[List[complex], int, bool]:
    r"""Core implementation of Aberth's root-finding algorithm.
    Iteratively improves root estimates using polynomial evaluations and derivative approximations.
    Convergence is achieved when all residuals fall below specified tolerance.

    The `aberth` function implements Aberth's method for polynomial root-finding. It works by:
    1. Evaluating the polynomial and its derivative at each current root estimate
    2. Adjusting each estimate based on the ratio of polynomial value to derivative
    3. Including correction terms from all other root estimates
    4. Repeating until convergence or maximum iterations reached

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

    .. svgbob::

                     P(zᵢ)
          zᵢ' = zᵢ - ──────
                     P'(zᵢ)

        where
                                   n
                                .-----.
                                 \      P(zᵢ)
          P'(zᵢ) = P₁(zᵢ) -       /    ────────
                                '-----' zᵢ - zⱼ
                                  j≠i

    Examples:
        >>> h = [5.0, 2.0, 9.0, 6.0, 2.0]
        >>> z0s = initial_aberth(h)
        >>> opt = Options()
        >>> opt.tolerance = 1e-8
        >>> zs, niter, found = aberth(h, z0s, opt)
        >>> found
        True

        >>> h = [1.0, -1.0, -1.0, 1.0]
        >>> z0s = initial_aberth(h)
        >>> zs, niter, found = aberth(h, z0s)
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


def initial_aberth_autocorr(coeffs: Sequence[float]) -> List[complex]:
    """
    Generates initial guesses for autocorrelation polynomials.
    Special case handling for polynomials with reciprocal root pairs.
    Adjusts radius to ensure roots stay within unit circle when possible.

    The function `initial_aberth_autocorr` calculates the initial values for the Aberth method for
    finding the roots of a polynomial. This version is specialized for autocorrelation polynomials,
    which have symmetric root structures (roots come in reciprocal conjugate pairs). It ensures
    the initial guesses are within the unit circle when possible.

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
    poly_c: Num = horner_eval_f(coeffs, center)
    radius: float | complex = pow(-poly_c, 1.0 / degree)
    # radius: float | complex = pow(-coeffs[-1], 1.0 / degree)
    if abs(radius) > 1.0:
        radius = 1.0 / radius
    c_gen = Circle(2)
    return [
        center + radius * complex(x, y)
        for y, x in (c_gen.pop() for _ in range(degree // 2))
    ]


def initial_aberth_autocorr_orig(coeffs: Sequence[float]) -> List[complex]:
    """
    Original trigonometric implementation for autocorrelation polynomials.
    Generates initial guesses on a circle with angular spacing considering reciprocal roots.
    Particularly suited for polynomials with symmetric root structures.

    The function `initial_aberth_autocorr_orig` calculates the initial guesses for the roots of a
    polynomial using the Aberth method. This version uses trigonometric functions to distribute
    the initial guesses and is specialized for autocorrelation polynomials, which have symmetric
    root structures (roots come in reciprocal conjugate pairs).

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
    poly_c: Num = horner_eval_f(coeffs, center)
    radius: float = pow(abs(poly_c), 1.0 / degree)
    # radius: float = pow(abs(coeffs[-1]), 1.0 / degree)
    if abs(radius) > 1:
        radius = 1 / radius
    degree //= 2
    k = 2.0 * math.pi / degree
    return [
        center + radius * (cos(theta) + sin(theta) * 1j)
        for theta in (k * (0.25 + i) for i in range(degree))
    ]


def aberth_autocorr(
    coeffs: Sequence[float], zs: List[complex], options: Options = Options()
) -> Tuple[List[complex], int, bool]:
    """
    Aberth's method variant for autocorrelation polynomials.
    Accounts for reciprocal root pairs (z and 1/z̄) in derivative calculation.
    Particularly useful for polynomials with symmetric coefficient structures.

    The `aberth_autocorr` function implements the Aberth method for finding the roots of a polynomial
    using autocorrelation. This version is specialized for polynomials where roots come in
    reciprocal conjugate pairs (common in signal processing applications). It modifies the
    standard Aberth method to account for these symmetric root structures.

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
             - List of refined roots
             - Number of iterations performed
             - Boolean indicating whether convergence was achieved

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
    coeffs: Sequence[float],
    i: int,
    zsc: List[complex],
) -> Tuple[float, int, complex]:
    """
    Worker function for multithreaded autocorrelation Aberth method.
    Handles individual root updates while considering reciprocal root pairs.
    Returns updated root estimate along with its residual for convergence checking.

    This function performs the core calculations for a single root in the multithreaded
    autocorrelation version of Aberth's method. It evaluates the polynomial and its
    derivative at the current root estimate, applies corrections for all other roots
    and their reciprocals, and returns the updated root estimate.

    :param coeffs: Polynomial coefficients in descending order of degree
    :param i: Index of the root being processed
    :param zsc: Current list of root estimates (complex numbers)
    :return: Tuple containing:
             - Residual (absolute value of polynomial at current estimate)
             - Index of the root being processed
             - New root estimate
    """
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
    coeffs: Sequence[float], zs: List[complex], options: Options = Options()
) -> Tuple[List[complex], int, bool]:
    """
    Multithreaded version of autocorrelation Aberth's method.
    Parallelizes root updates across multiple threads for improved performance.
    Maintains thread safety by keeping root updates in separate jobs.

    This function implements the autocorrelation version of Aberth's method using multiple
    threads. Each root update is performed in parallel, which can significantly speed up
    computation for high-degree polynomials. The function maintains the same mathematical
    operations as the single-threaded version but distributes the workload across available
    processors.

    :param coeffs: List of polynomial coefficients in descending order of degree
    :param zs: Initial guesses for the roots (complex numbers)
    :param options: Configuration options including max iterations and tolerance
    :return: Tuple containing:
             - List of refined roots
             - Number of iterations performed
             - Boolean indicating whether convergence was achieved
    """
    with ThreadPoolExecutor() as executor:
        for niter in range(options.max_iters):
            tolerance = 0.0
            futures = []

            for i in range(len(zs)):
                futures.append(executor.submit(aberth_autocorr_job, coeffs, i, zs[:]))

            for future in futures:
                tol_i, i, zi = future.result()
                if tol_i > tolerance:
                    tolerance = tol_i
                zs[i] = zi

            if tolerance < options.tolerance:
                return zs, niter, True

    return zs, options.max_iters, False
