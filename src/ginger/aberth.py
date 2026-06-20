"""
Aberth-Ehrlich method for simultaneous polynomial root-finding.

Aberth's method (Ehrlich, 1967; Aberth, 1973) combines Newton's method with an
implicit deflation strategy to compute all roots of a polynomial simultaneously
with cubic convergence. The correction formula is:

    zᵢ' = zᵢ - P(zᵢ) / P'(zᵢ)

    P'(zᵢ) = P₁(zᵢ) - Σⱼ₌ᵢ P(zᵢ) / (zᵢ - zⱼ)

where P₁(z) is the derivative of P(z).

This module provides:
    aberth              — single-threaded Aberth method
    aberth_mt           — multithreaded version
    aberth_autocorr     — variant for autocorrelation (palindromic) polynomials
    initial_aberth      — LDS-based initial guess generation
    poly_from_roots     — reconstruct polynomial from roots (Leja ordering)
"""

import math
from concurrent.futures import ThreadPoolExecutor
from math import cos, pi, sin
from typing import List, Sequence, Tuple, Union

from lds_gen.lds import TWO_PI, VdCorput

from .rootfinding import Options, horner_eval, horner_eval_f

Num = Union[float, complex]
# from mywheel.robin import Robin

# ---------------------------------------------------------------------------
# Precomputed LDS tables (replaces runtime Circle/VdCorput generation)
# ---------------------------------------------------------------------------
TABLE_SIZE = 1000

# VdCorput<2> sequence table (index 0 = first pop)
_vgen = VdCorput(2)
VDC_TABLE_2: List[float] = [_vgen.pop() for _ in range(TABLE_SIZE)]

# Circle<2> table: each entry is (cos(theta), sin(theta))
CIRCLE_TABLE_2: List[Tuple[float, float]] = [
    (cos(v * TWO_PI), sin(v * TWO_PI)) for v in VDC_TABLE_2
]

# cos(pi * vdc) table (used by Bairstow initial_guess)
COS_PI_VDC2_TABLE: List[float] = [cos(pi * v) for v in VDC_TABLE_2]


def horner_backward(coeffs1: List, degree: int, alpha: complex) -> complex:
    """
    Backward Horner evaluation for root refinement.

    Evaluates polynomial at x = α using coefficients in reverse order,
    modifying them in-place to center around α.

    :param coeffs1: Polynomial coefficients in descending order (modified in-place)
    :param degree: Degree of the polynomial
    :param alpha: Center point for evaluation
    :return: Value of the polynomial at α

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
    r"""Generate initial root guesses using a low-discrepancy sequence.

    The center :math:`c` and radius :math:`R` are derived from the
    coefficients:

    .. math::

       c &= -\frac{a_1}{n a_0} \\[4pt]
       R &= \sqrt[n]{-P(c)} \\[4pt]
       z_i &= c + R \cdot (\cos\theta_i + i\sin\theta_i)

    where :math:`\theta_i = 2\pi v_i` with :math:`v_i` from a van der
    Corput sequence for low-discrepancy angular spacing.

    :param coeffs: Polynomial coefficients in descending order
    :return: Initial root guesses as complex numbers

    Examples:
        >>> h = [5.0, 2.0, 9.0, 6.0, 2.0]
        >>> z0s = initial_aberth(h)
    """
    degree: int = len(coeffs) - 1
    center: float = -coeffs[1] / (degree * coeffs[0])
    poly_c: Num = horner_eval_f(coeffs, center)
    radius: float | complex = pow(-poly_c, 1.0 / degree)
    # radius: float = pow(abs(poly_c), 1.0 / degree)
    return [
        center + radius * complex(x, y)
        for y, x in (CIRCLE_TABLE_2[i] for i in range(degree))
        #    ^------ Note! swap y and x
    ]


def initial_aberth_orig(coeffs: Sequence[float]) -> List[complex]:
    """
    Original trigonometric initial guess generator.

    Places roots equally spaced around a circle with angular offset of 0.25
    radians to avoid alignment with coordinate axes.

    :param coeffs: Polynomial coefficients in descending order
    :return: Initial root guesses as complex numbers

    Examples:
        >>> h = [5.0, 2.0, 9.0, 6.0, 2.0]
        >>> z0s = initial_aberth_orig(h)
    """
    degree: int = len(coeffs) - 1
    center: float = -coeffs[1] / (degree * coeffs[0])
    poly_c: Num = horner_eval_f(coeffs, center)
    radius: float | complex = pow(-poly_c, 1.0 / degree)
    angle_step = 2.0 * math.pi / degree
    return [
        center + radius * (cos(theta) + sin(theta) * 1j)
        for theta in (angle_step * (0.25 + i) for i in range(degree))
    ]


def aberth_mt(
    coeffs: Sequence[float], zs: List[complex], options: Options = Options()
) -> Tuple[List[complex], int, bool]:
    """
    Multithreaded Aberth method using ThreadPoolExecutor.

    Parallelizes root updates across available CPUs while checking convergence
    in the main thread.

    :param coeffs: Polynomial coefficients in descending order
    :param zs: Initial root guesses
    :param options: Algorithm configuration
    :return: Tuple of (refined roots, iterations, converged)
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
    r"""Aberth-Ehrlich simultaneous root-finding (single-threaded).

    Combines Newton's method with an implicit deflation strategy. Each root
    estimate :math:`z_i` is updated by the correction:

    .. math::

       z_i' = z_i - \frac{P(z_i)}{P'(z_i)}

    where the derivative is modified by all other root estimates:

    .. math::

       P'(z_i) = P_1(z_i) - \sum_{\substack{j=1\\j\neq i}}^n
                 \frac{P(z_i)}{z_i - z_j}

    Here :math:`P_1(z)` is the ordinary derivative of :math:`P(z)`.
    Convergence is cubic.

    :param coeffs: Polynomial coefficients in descending order
    :param zs: Initial root guesses
    :param options: Algorithm configuration
    :return: Tuple of (final roots, iterations performed, converged)

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
    Generate initial guesses for autocorrelation (palindromic) polynomials.

    Adjusts radius to keep initial guesses within the unit circle, matching
    the reciprocal-root structure of autocorrelation polynomials.

    :param coeffs: Polynomial coefficients in descending order
    :return: Initial root guesses

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
    return [
        center + radius * complex(x, y)
        for y, x in (CIRCLE_TABLE_2[i] for i in range(degree // 2))
    ]


def initial_aberth_autocorr_orig(coeffs: Sequence[float]) -> List[complex]:
    """
    Original trigonometric initial guess generator for autocorrelation polynomials.

    :param coeffs: Polynomial coefficients in descending order
    :return: Initial root guesses

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
    angle_step = 2.0 * math.pi / degree
    return [
        center + radius * (cos(theta) + sin(theta) * 1j)
        for theta in (angle_step * (0.25 + i) for i in range(degree))
    ]


def aberth_autocorr(
    coeffs: Sequence[float], zs: List[complex], options: Options = Options()
) -> Tuple[List[complex], int, bool]:
    r"""Aberth method for palindromic (autocorrelation) polynomials.

    Same as :func:`aberth` but additionally accounts for reciprocal root pairs
    :math:`z` and :math:`1/\bar{z}` in the correction:

    .. math::

       P'(z_i) = P_1(z_i) - \sum_{\substack{j=1\\j\neq i}}^n
                 \left(\frac{P(z_i)}{z_i - z_j} +
                       \frac{P(z_i)}{z_i - 1/z_j}\right)

    This preserves the reciprocal-root structure of autocorrelation
    (palindromic) polynomials.

    :param coeffs: Polynomial coefficients in descending order
    :param zs: Initial root guesses
    :param options: Algorithm configuration
    :return: Tuple of (refined roots, iterations, converged)

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
    Worker for multithreaded autocorrelation Aberth — updates a single root.

    Evaluates polynomial and derivative at zi, applies corrections from all
    other roots and their reciprocals, returns the updated value.

    :param coeffs: Polynomial coefficients in descending order
    :param i: Index of the root to update
    :param zsc: Current list of root estimates
    :return: Tuple of (residual at zi, root index, new root estimate)
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
    Multithreaded autocorrelation Aberth method.

    Parallelizes root updates across threads, accounting for reciprocal root
    pairs in the correction step.

    :param coeffs: Polynomial coefficients in descending order
    :param zs: Initial root guesses
    :param options: Algorithm configuration
    :return: Tuple of (refined roots, iterations, converged)
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


def leja_order(points: List[complex]) -> List[complex]:
    r"""Greedy Leja ordering for numerical stability.

    Starts with :math:`p_0 = \arg\min |p|`, then iteratively selects:

    .. math::

       p_k = \arg\max_{p \in \text{remaining}}
             \min_{q \in \text{selected}} |p - q|

    This maximizes the minimum distance between successive points,
    improving numerical stability in polynomial reconstruction.

    :param points: Input complex points
    :return: Reordered points in Leja sequence

    Examples:
        >>> pts = [1+0j, -1+0j, 0+1j, 0-1j]
        >>> ordered = leja_order(pts)
        >>> len(ordered)
        4
    """
    if not points:
        return []
    sorted_pts = sorted(points, key=abs)
    result = [sorted_pts[0]]
    remaining = sorted_pts[1:]
    while remaining:
        best_idx = 0
        best_dist = -1.0
        for i, p in enumerate(remaining):
            min_dist = min(abs(p - q) for q in result)
            if min_dist > best_dist:
                best_dist = min_dist
                best_idx = i
        result.append(remaining.pop(best_idx))
    return result


def poly_from_roots(zs: List[complex]) -> List[float]:
    r"""Reconstruct a monic polynomial from its roots.

    Orders roots with Leja ordering for stability, then convolves factors:

    .. math::

       P(x) = \prod_{i=1}^n (x - r_i)
            = x^n + a_{n-1} x^{n-1} + \cdots + a_0

    The coefficients are computed by repeated convolution, i.e.:
    starting from :math:`[1]`, for each root :math:`r`:

    .. math::

       c_{k+1} \leftarrow c_{k+1} - r \cdot c_k

    :param zs: Roots of the polynomial
    :return: Monic polynomial coefficients (highest degree first)

    Examples:
        >>> coeffs = poly_from_roots([1+0j, -1+0j])
        >>> coeffs
        [1.0, 0.0, -1.0]
    """
    ordered = leja_order(zs)
    coeffs = [1.0 + 0.0j]
    for z in ordered:
        prev = coeffs[0]
        for i in range(1, len(coeffs)):
            old = coeffs[i]
            coeffs[i] -= z * prev
            prev = old
        coeffs.append(-z * prev)
    return [c.real for c in coeffs]


def poly_from_autocorr_roots(zs: List[complex]) -> List[float]:
    """
    Reconstruct a monic polynomial from autocorrelation roots.

    Palindromic polynomials have roots in reciprocal pairs. This function
    adds 1/z for each root, then reconstructs via :func:`poly_from_roots`.

    :param zs: Roots from aberth_autocorr or aberth_autocorr_mt
    :return: Monic polynomial coefficients (highest degree first)

    Examples:
        >>> zs = [0.5+0.5j, 0.5-0.5j]
        >>> coeffs = poly_from_autocorr_roots(zs)
        >>> len(coeffs)
        5
    """
    if not zs:
        return [1.0]
    all_roots: List[complex] = []
    for z in zs:
        all_roots.append(z)
        all_roots.append(1.0 / z)
    return poly_from_roots(all_roots)
