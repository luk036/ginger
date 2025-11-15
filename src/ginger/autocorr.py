from math import cos, pi, sqrt
from typing import List

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
    k = pi / degree  # Angular step size between roots

    m = radius * radius  # Quadratic term for Vector2
    # Generate initial guesses using cosine distribution of roots
    return [Vector2(2 * radius * cos(k * i), -m) for i in range(1, degree, 2)]


def pbairstow_autocorr(
    coeffs: List[float], vrs: List[Vector2], options: Options = Options()
):
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
    M = len(vrs)  # Number of quadratic factors
    degree = len(coeffs) - 1
    converged = [False] * M  # Convergence status tracker
    robin = Robin(M)  # Round-robin iterator for factor updates

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
    r, q = vr.x, vr.y
    hr = r / 2.0  # Half-radius
    d = hr * hr + q  # Discriminant

    if d < 0.0:  # Complex conjugate roots case
        if q < -1.0:  # Ensure magnitude using reciprocal
            vr = Vector2(-r, 1.0) / q
    else:  # Real roots case
        # Calculate roots using alternative quadratic formula
        a1 = hr + (sqrt(d) if hr >= 0.0 else -sqrt(d))
        a2 = -q / a1  # Second root from Vieta's formula

        # Handle roots outside unit circle
        if abs(a1) > 1.0:
            a1 = 1.0 / a1
            if abs(a2) > 1.0:
                a2 = 1.0 / a2
            vr = Vector2(a1 + a2, -a1 * a2)
        elif abs(a2) > 1.0:
            a2 = 1.0 / a2
            vr = Vector2(a1 + a2, -a1 * a2)

    return vr
