"""
Parallel Bairstow root-finding for real-coefficient polynomials.

This module implements a parallel version of Bairstow's method for finding all
roots of a polynomial simultaneously. The algorithm factors the polynomial into
quadratic factors (x² - r·x - q), using suppression to decouple root estimates,
enabling parallel refinement.

Key functions:
    pbairstow_even  — main parallel Bairstow solver
    initial_guess   — generate starting estimates from coefficients
    suppress        — decouple interference between root factors
    horner          — evaluate polynomial against quadratic factor
    roots_from_quadratic — extract two roots from a quadratic factor
"""

from functools import reduce
from itertools import accumulate
from math import sqrt
from typing import Any, List, Sequence, Tuple, Union

from mywheel.robin import Robin

from .matrix2 import Matrix2
from .vector2 import Vector2

Num = Union[float, complex]


class Options:
    """Configuration parameters for iterative root-finding algorithms.

    Attributes:
        max_iters: Maximum number of iterations (default 2000)
        tolerance: Global convergence tolerance (default 1e-12)
        tol_ind: Per-root convergence tolerance (default 1e-15)
    """

    max_iters: int = 2000
    tolerance: float = 1e-12
    tol_ind: float = 1e-15
    # tol_suppress: float = 1e-1


def delta(vA: Vector2, vr: Vector2, vp: Vector2) -> Vector2:
    r"""Calculate adjustment vector for Bairstow's method.

    Solves the 2×2 linear system to find the optimal adjustment to current
    quadratic factor estimates :math:`(r,q)`:

    .. math::

       \begin{bmatrix} r p + s & p \\ q p & s \end{bmatrix}
       \begin{bmatrix} \Delta r \\ \Delta q \end{bmatrix}
       = \begin{bmatrix} A \\ B \end{bmatrix}

    where :math:`(p,s) = (r_i - r_j,\; q_i - q_j)` is the difference
    between two factor estimates, and :math:`(A,B)` is the remainder from
    polynomial division.

    :param vA: Residual vector :math:`(A,B)` from polynomial division
    :param vr: Current root estimate :math:`(r,q)`
    :param vp: Suppression vector :math:`(p,s) = \mathbf{vr}_i - \mathbf{vr}_j`
    :return: Correction vector :math:`(\Delta r, \Delta q)`

    Examples:
        >>> d = delta(Vector2(1, 2), Vector2(-2, 0), Vector2(4, 5))
        >>> print(d)
        <0.2, 0.4>
    """
    r_coeff, q_coeff = vr.x, vr.y
    p_coeff, s_coeff = vp.x, vp.y
    mp = Matrix2(
        Vector2(s_coeff, -p_coeff),
        Vector2(-p_coeff * q_coeff, p_coeff * r_coeff + s_coeff),
    )
    return mp.mdot(vA) / mp.det()  # 6 mul's + 2 div's


def suppress_old(vA: Vector2, vA1: Vector2, vri: Vector2, vrj: Vector2) -> None:
    """Original zero suppression for Bairstow's method (modifies in-place).

    Modifies residual vectors vA and vA1 to suppress interference from other
    root estimates (vrj) during iteration.

    Note: This original version modifies vectors in-place. The newer
    :func:`suppress` returns modified copies instead.

    :param vA: Current residual vector (A,B)
    :param vA1: First derivative residual vector (A1,B1)
    :param vri: Current root estimate (ri,qi)
    :param vrj: Other root estimate (rj,qj)

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
    A_val, B_val = vA.x, vA.y
    A1_val, B1_val = vA1.x, vA1.y
    vp = vri - vrj
    r_coeff, q_coeff = vri.x, vri.y
    p_coeff, s_coeff = vp.x, vp.y
    f_val = r_coeff * p_coeff + s_coeff
    qp_val = q_coeff * p_coeff
    e_val = f_val * s_coeff - qp_val * p_coeff
    a_val = A_val * s_coeff - B_val * p_coeff
    b_val = B_val * f_val - A_val * qp_val
    c_val = A1_val * e_val - a_val
    d_val = B1_val * e_val - b_val - a_val * p_coeff
    vA._x = a_val * e_val
    vA._y = b_val * e_val
    vA1._x = c_val * s_coeff - d_val * p_coeff
    vA1._y = d_val * f_val - c_val * qp_val
    # return delta(vA, vri, Vector2(vA1._x, -vA1._y))


def suppress(
    vA: Vector2, vA1: Vector2, vri: Vector2, vrj: Vector2
) -> Tuple[Vector2, Vector2]:
    r"""Zero-suppression for decoupling Bairstow factor estimates.

    Modifies the residual vectors :math:`(A,B)` and :math:`(A_1,B_1)` to
    suppress interference from another quadratic factor :math:`(r_j, q_j)`.
    The adjugate matrix is:

    .. math::

       \mathbf{M}^* =
       \begin{bmatrix} s & -p \\ -p q_i & p r_i + s \end{bmatrix},
       \qquad \det(\mathbf{M}^*) = s(p r_i + s) - p^2 q_i

    where :math:`(p,s) = (r_i - r_j,\; q_i - q_j)`. The modified
    residuals are:

    .. math::

       \begin{bmatrix} A' \\ B' \end{bmatrix}
       &= \det(\mathbf{M}^*) \cdot \mathbf{M}^*
          \begin{bmatrix} A \\ B \end{bmatrix} \\[4pt]
       \begin{bmatrix} A'_1 \\ B'_1 \end{bmatrix}
       &= \mathbf{M}^*\bigl(
          \mathbf{M}^* \begin{bmatrix} A_1 \\ B_1 \end{bmatrix}
          - \begin{bmatrix} A' \\ B' \end{bmatrix}
          \bigr)

    Reference:
        D. C. Handscomb, *Computer Journal*, 5 (1962), pp. 139-141.

    :param vA: Current residual :math:`(A,B)`
    :param vA1: Derivative residual :math:`(A_1,B_1)`
    :param vri: Current factor :math:`(r_i,q_i)`
    :param vrj: Other factor :math:`(r_j,q_j)`
    :return: Modified residuals :math:`(A',B')` and :math:`(A'_1,B'_1)`

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
    r_coeff, q_coeff = vri.x, vri.y
    p_coeff, s_coeff = vp.x, vp.y
    m_adjoint = Matrix2(
        Vector2(s_coeff, -p_coeff),
        Vector2(-p_coeff * q_coeff, p_coeff * r_coeff + s_coeff),
    )
    e_val = m_adjoint.det()
    va = m_adjoint.mdot(vA)
    vc = vA1 * e_val - va
    vc._y -= va._x * p_coeff
    va *= e_val
    va1 = m_adjoint.mdot(vc)
    return va, va1


def horner_eval_f(coeffs: Sequence[Num], zval: Num) -> Num:
    """Evaluate polynomial using Horner's method (functional reduce version).

    :param coeffs: Polynomial coefficients in descending order of degree
    :param zval: Point at which to evaluate the polynomial
    :return: Value of the polynomial at zval

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
def horner_eval(coeffs: Sequence[Num], zval: Num) -> Tuple[Any, List[Num]]:
    """Evaluate polynomial via Horner, returning value and intermediate coefficients.

    :param coeffs: Polynomial coefficients in descending order of degree
    :param zval: Point at which to evaluate the polynomial
    :return: Tuple of (polynomial value at zval, intermediate coefficients)

    Examples:
        >>> coeffs = [1, -8, -72, 382, 727, -2310]
        >>> horner_eval(coeffs, 3)
        (960, [1, -5, -87, 121, 1090, 960])
        >>> horner_eval(coeffs, 3+0j)
        ((960+0j), [1, (-5+0j), (-87+0j), (121+0j), (1090+0j), (960+0j)])
    """
    coeffs = list(accumulate(coeffs, lambda res, coeff: res * zval + coeff))
    return coeffs[-1], coeffs


#
#                       ⎛ 2            ⎞
#        P(x) = P (x) ⋅ ⎝x  - r ⋅ x - q⎠ + A ⋅ x + B
#                1
#
#    Note: P(x) becomes the quotient after calling this function
def horner(coeffs: List[float], degree: int, vr: Vector2) -> Vector2:
    r"""Synthetic division by a quadratic factor :math:`x^2 - r x - q`.

    Performs polynomial division:

    .. math::

       P(x) = Q(x)\cdot(x^2 - r x - q) + A x + B

    Returns the linear remainder :math:`(A,B)` and overwrites ``coeffs``
    with the quotient :math:`Q(x)` (deflated polynomial).

    The recurrence for the quotient coefficients :math:`b_k` is:

    .. math::

       b_0 &= a_0 \\
       b_1 &= a_1 + r b_0 \\
       b_k &= a_k + r b_{k-1} + q b_{k-2} \quad (k \ge 2)

    with remainder :math:`A = b_{n-1},\; B = b_n + q b_{n-1}`.

    :param coeffs: Polynomial coefficients in descending order (modified in-place)
    :param degree: Degree of the polynomial (must be ≥ 2)
    :param vr: Quadratic factor :math:`(r,q)`
    :return: Remainder :math:`(A,B)`

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


def initial_guess(coeffs: List[float]) -> List[Vector2]:
    r"""Generate initial quadratic-factor estimates for Bairstow's method.

    Estimates are placed around a circle centered at :math:`c` with radius
    :math:`R`:

    .. math::

       c &= -\frac{a_1}{n a_0},\qquad
       R = \sqrt[n]{|P(c)|} \\[4pt]
       \theta_k &= \pi \cdot \text{VDC}_2[k+1] \\[4pt]
       (r_k, q_k) &= \bigl(2(c + R\cos\theta_k),\;
                     -(c^2 + R^2 + 2cR\cos\theta_k)\bigr)

    where :math:`\text{VDC}_2` is a van der Corput low-discrepancy sequence.

    :param coeffs: Polynomial coefficients in descending order
    :return: List of initial quadratic factors as :class:`~ginger.vector2.Vector2` :math:`(r,q)`

    Examples:
        >>> h = [10.0, 34.0, 75.0, 94.0, 150.0, 94.0, 75.0, 34.0, 10.0]
        >>> vr0s = initial_guess(h)
    """
    degree: int = len(coeffs) - 1
    center: float = -coeffs[1] / (degree * coeffs[0])
    poly_c: Num = horner_eval_f(coeffs, center)
    radius: float = pow(abs(poly_c), 1.0 / degree)
    quad_term: float = center * center + radius * radius
    degree //= 2
    degree *= 2  # make even

    from .aberth import COS_PI_VDC2_TABLE

    temp = iter(radius * COS_PI_VDC2_TABLE[i + 1] for i in range(0, degree // 2))
    return [Vector2(2 * (center + t), -(quad_term + 2 * center * t)) for t in temp]


def pbairstow_even(
    coeffs: List[float], vrs: List[Vector2], options: Options = Options()
) -> Tuple[List[Vector2], int, bool]:
    r"""Parallel Bairstow method for simultaneous root-finding.

    Iteratively refines :math:`m = n/2` quadratic factors of the polynomial
    using suppression to decouple estimates. Each factor :math:`(r_i, q_i)`
    is updated by solving a 2×2 system:

    .. math::

       \begin{bmatrix} r_i \\ q_i \end{bmatrix}^{\!(k+1)}
       = \begin{bmatrix} r_i \\ q_i \end{bmatrix}^{\!(k)}
       - \begin{bmatrix}
          A'_1 r_i + B'_1 & A'_1 \\
          A'_1 q_i        & B'_1
          \end{bmatrix}^{-1}
         \begin{bmatrix} A \\ B \end{bmatrix}

    where the "suppressed" derivative residuals are:

    .. math::

       \begin{bmatrix} A'_1 \\ B'_1 \end{bmatrix}
       = \begin{bmatrix} A_1 \\ B_1 \end{bmatrix}
       - \sum_{j \neq i}
         \begin{bmatrix}
         p_{ij} r_i + s_{ij} & p_{ij} \\
         p_{ij} q_i          & s_{ij}
         \end{bmatrix}^{-1}
         \begin{bmatrix} A \\ B \end{bmatrix}

    with :math:`(p_{ij}, s_{ij}) = (r_i - r_j,\; q_i - q_j)`.

    :param coeffs: Polynomial coefficients in descending order
    :param vrs: Initial estimates for quadratic factors
    :param options: Algorithm configuration parameters
    :return: Tuple of (final root estimates, iterations performed, converged)

    Examples:
        >>> h = [10.0, 34.0, 75.0, 94.0, 150.0, 94.0, 75.0, 34.0, 10.0]
        >>> vr0s = initial_guess(h)
        >>> vrs, niter, found = pbairstow_even(h, vr0s)
        >>> print(found)
        True
    """
    num_factors = len(vrs)
    degree = len(coeffs) - 1
    converged = [False] * num_factors
    robin = Robin(num_factors)
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


def roots_from_quadratic(vr: Vector2) -> Tuple[complex, complex]:
    r"""Solve :math:`x^2 - r x - q = 0` for its two roots.

    Uses the quadratic formula:

    .. math::

       x = \frac{r \pm \sqrt{r^2 + 4q}}{2}

    :param vr: Vector2 with :math:`x=r, y=q` representing :math:`x^2 - r x - q`
    :return: The two roots (real or complex conjugate pair)

    Examples:
        >>> r1, r2 = roots_from_quadratic(Vector2(0, 1))  # x^2 - 1
        >>> print(r1, r2)
        1.0 -1.0
        >>> r1, r2 = roots_from_quadratic(Vector2(0, -1))  # x^2 + 1
        >>> abs(r1 - 1j) < 1e-14
        True
        >>> abs(r2 + 1j) < 1e-14
        True
    """
    r = vr.x
    q = vr.y
    disc = r * r + 4.0 * q
    if disc >= 0.0:
        sqrt_disc = sqrt(disc)
        return ((r + sqrt_disc) / 2.0, (r - sqrt_disc) / 2.0)
    sqrt_disc = sqrt(-disc)
    return (complex(r / 2.0, sqrt_disc / 2.0), complex(r / 2.0, -sqrt_disc / 2.0))


def poly_from_quadratic_factors(vrs: List[Vector2]) -> List[float]:
    """
    Reconstruct a monic polynomial from its quadratic factors.

    Each factor x² - r·x - q is converted to its two roots, and the full
    polynomial is reconstructed via :func:`aberth.poly_from_roots` with
    Leja ordering for numerical stability.

    :param vrs: Quadratic factors from Bairstow's method
    :return: Monic polynomial coefficients (highest degree first)

    Examples:
        >>> vrs = [Vector2(0.0, 1.0)]  # x^2 - 1
        >>> coeffs = poly_from_quadratic_factors(vrs)
        >>> coeffs
        [1.0, 0.0, -1.0]
    """
    if not vrs:
        return [1.0]
    from .aberth import poly_from_roots

    all_roots: List[complex] = []
    for vr in vrs:
        r1, r2 = roots_from_quadratic(vr)
        all_roots.append(r1)
        all_roots.append(r2)
    return poly_from_roots(all_roots)


def find_rootq(vr: Vector2) -> Tuple[Num, Num]:
    r"""Solve :math:`x^2 - r x - q = 0` using Vieta's formula.

    Uses the alternative (Citardauq) formula to avoid catastrophic
    cancellation when :math:`|r|` is large:

    .. math::

       x_1 &= \frac{r}{2} + \operatorname{sgn}\!\left(\frac{r}{2}\right)
             \sqrt{\left(\frac{r}{2}\right)^2 + q} \\[4pt]
       x_2 &= -\frac{q}{x_1} \qquad\text{(Vieta's formula)}

    :param vr: Quadratic coefficients :math:`(r,q)`
    :return: The two roots

    Examples:
        >>> vr = find_rootq(Vector2(5, -6))
        >>> print(vr)
        (3.0, 2.0)
    """

    half_radius = vr.x / 2
    discriminant = half_radius * half_radius + vr.y
    if discriminant < 0:
        root1 = half_radius + sqrt(-discriminant) * 1j
    else:
        root1 = half_radius + (
            sqrt(discriminant) if half_radius >= 0 else -sqrt(discriminant)
        )
    root2 = -vr.y / root1
    return root1, root2
