from math import cos, pi, sqrt
from typing import List, Tuple

from mywheel.robin import Robin

from .rootfinding import Options, delta, horner, suppress
from .vector2 import Vector2


def initial_autocorr(coeffs: List[float]) -> List[Vector2]:
    """
    Calculates initial guesses for autocorrelation roots using coefficient analysis.

    The method:
    1. Computes a root radius estimate from the constant term
    2. Adjusts radius to focus on roots outside unit circle
    3. Generates initial guesses using cosine spaced angles with quadratic terms

    :param coeffs: Polynomial coefficients from highest to lowest degree
    :return: List of Vector2 representing quadratic factors (x² - rx - q)

    Examples:
        >>> h = [10.0, 34.0, 75.0, 94.0, 150.0, 94.0, 75.0, 34.0, 10.0]
        >>> vrs = initial_autocorr(h)
    """
    degree = len(coeffs) - 1
    # Calculate initial radius estimate using absolute value of constant term
    radius = pow(abs(coeffs[-1]), 1.0 / degree)
    if radius < 1:  # Focus on roots outside unit circle by taking reciprocal
        radius = 1 / radius
    degree //= 2  # Work with half-degree for conjugate pairs
    angle_step = pi / degree  # Angular step size between roots

    quad_term = radius * radius  # Quadratic term for Vector2
    # Generate initial guesses using cosine distribution of roots
    return [
        Vector2(2 * radius * cos(angle_step * i), -quad_term)
        for i in range(1, degree, 2)
    ]


def pbairstow_autocorr(
    coeffs: List[float], vrs: List[Vector2], options: Options = Options()
) -> Tuple[List[Vector2], int, bool]:
    """
    Implements Bairstow's method for polynomial root finding with autocorrelation.

    Process outline:
    1. Iterates until convergence or max iterations
    2. Evaluates polynomial and first derivative using Horner's scheme
    3. Suppresses interference from other roots
    4. Updates estimates using Newton-Raphson step

    :param coeffs: Polynomial coefficients (degree must be even)
    :param vrs: Initial guesses for quadratic factors
    :param options: Algorithm control parameters
    :return: Tuple of (updated factors, iterations, convergence status)

    Examples:
        >>> h = [10.0, 34.0, 75.0, 94.0, 150.0, 94.0, 75.0, 34.0, 10.0]
        >>> vrs = initial_autocorr(h)
        >>> vrs, niter, found = pbairstow_autocorr(h, vrs)
        >>> found
        True
    """
    num_factors = len(vrs)  # Number of quadratic factors
    degree = len(coeffs) - 1
    converged = [False] * num_factors  # Convergence status tracker
    robin = Robin(num_factors)  # Round-robin iterator for factor updates

    for niter in range(options.max_iters):
        tolerance = 0.0  # Track maximum error across all factors
        for i, (vri, ci) in enumerate(zip(vrs, converged)):
            if ci:  # Skip already converged factors
                continue

            # Polynomial evaluation at current factor estimate
            coeffs1 = coeffs.copy()
            vA = horner(coeffs1, degree, vri)

            # Check individual convergence
            tol_i = max(abs(vA.x), abs(vA.y))
            if tol_i < options.tol_ind:
                converged[i] = True
                continue
            tolerance = max(tolerance, tol_i)

            # Evaluate reduced polynomial (degree-2)
            vA1 = horner(coeffs1, degree - 2, vri)

            # Suppress influence of other factors
            for j in robin.exclude(i):
                vrj = vrs[j]
                vA, vA1 = suppress(vA, vA1, vri, vrj)
                # Handle reciprocal roots
                vrn = Vector2(-vrj.x, 1.0) / vrj.y
                vA, vA1 = suppress(vA, vA1, vri, vrn)

            # Apply Newton-Raphson update
            vrs[i] -= delta(vA, vri, vA1)

        # Check global convergence
        if tolerance < options.tolerance:
            return vrs, niter, True

    return vrs, options.max_iters, False


def poly_from_autocorr_factors(vrs: List[Vector2]) -> List[float]:
    """
    Reconstruct a monic polynomial from autocorrelation quadratic factors.

    Each quadratic factor x^2 - r*x - q contributes 2 roots. For palindromic
    polynomials, the reciprocal of each root is also a root. This function
    extracts all roots, adds reciprocals, then reconstructs with Leja ordering.

    :param vrs: Quadratic factors from pbairstow_autocorr
    :return: Monic polynomial coefficients (highest degree first)

    Examples:
        >>> vrs = [Vector2(0.0, 1.0)]  # x^2 - 1 => roots 1, -1 => +reciprocals
        >>> coeffs = poly_from_autocorr_factors(vrs)
        >>> len(coeffs)
        5
    """
    if not vrs:
        return [1.0]
    from .aberth import poly_from_roots
    from .rootfinding import roots_from_quadratic

    all_roots: List[complex] = []
    for vr in vrs:
        r1, r2 = roots_from_quadratic(vr)
        all_roots.append(r1)
        all_roots.append(r2)
        all_roots.append(1.0 / r1)
        all_roots.append(1.0 / r2)
    return poly_from_roots(all_roots)


def extract_autocorr(vr: Vector2) -> Vector2:
    """
    Normalizes quadratic factors to ensure roots within unit circle.

    Strategy:
    1. Calculate roots of quadratic x² - rx - q
    2. If roots are outside unit circle, take reciprocals
    3. Return new quadratic with roots inside unit circle

    :param vr: Vector2 representing quadratic coefficients (r, q)
    :return: Normalized Vector2 with roots inside unit circle

    Examples:
        >>> vr = Vector2(5, -6)
        >>> vr_new = extract_autocorr(vr)
        >>> print(vr_new)
        <0.8333333333333333, -0.16666666666666666>
    """
    r_coeff, q_coeff = vr.x, vr.y
    half_radius = r_coeff / 2.0  # Half-radius
    discriminant = half_radius * half_radius + q_coeff  # Discriminant

    if discriminant < 0.0:  # Complex conjugate roots case
        if q_coeff < -1.0:  # Ensure magnitude using reciprocal
            vr = Vector2(-r_coeff, 1.0) / q_coeff
    else:  # Real roots case
        # Calculate roots using alternative quadratic formula
        root1 = half_radius + (
            sqrt(discriminant) if half_radius >= 0.0 else -sqrt(discriminant)
        )
        root2 = -q_coeff / root1  # Second root from Vieta's formula

        # Handle roots outside unit circle
        if abs(root1) > 1.0:
            root1 = 1.0 / root1
            if abs(root2) > 1.0:
                root2 = 1.0 / root2
            vr = Vector2(root1 + root2, -root1 * root2)
        elif abs(root2) > 1.0:
            root2 = 1.0 / root2
            vr = Vector2(root1 + root2, -root1 * root2)

    return vr
