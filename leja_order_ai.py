"""
Leja Order

This code implements algorithms related to the Leja ordering of points and polynomial factorization. It consists of three main functions: leja_order, seprts, and sfact. Let's break down what each part does in simple terms.

The leja_order function takes a set of complex points as input and rearranges them in a special order called the Leja order. This ordering is useful in numerical methods, especially for tasks like polynomial interpolation. The function starts by selecting the point with the smallest magnitude, then repeatedly chooses the next point that is farthest from the last selected point. This process continues until all points are ordered. The output is a numpy array of the input points in their new Leja order.

The seprts function takes the coefficients of a polynomial as input and separates its roots (the values that make the polynomial equal to zero) into three categories: those inside a circle with radius 1, those on this circle, and those outside it. It focuses on the roots on the circle, sorts them by their angle, and selects every other one. The function returns these selected roots along with the roots inside the circle.

Lastly, the sfact function performs what's called a Schur-Cohn factorization on a polynomial. It uses the seprts function to find the roots, then applies the leja_order to these roots. It creates a new polynomial from these Leja-ordered roots and normalizes it. The function returns this normalized polynomial (called the Schur-Cohn factor) along with the Leja-ordered roots.

The code achieves its purpose by using mathematical concepts like complex numbers, polynomials, and geometric principles. It transforms the input data (points or polynomial coefficients) through a series of steps involving calculations of distances, angles, and polynomial operations.

An important aspect of the code's logic is how it iteratively builds the Leja order in the leja_order function. It repeatedly calculates distances and selects points based on these distances, gradually constructing the desired ordering.

Overall, this code provides tools for working with complex points and polynomials, particularly in ways that can improve numerical stability in certain mathematical computations. It takes in either a set of points or polynomial coefficients and produces ordered points or factorized polynomials that have desirable properties for further calculations.
"""

import numpy as np
from typing import List


def leja_order(points):
    """
    Compute the Leja order of a set of complex points.

    The Leja order is a way to order a set of complex points such that the minimum distance between consecutive points is maximized. This is useful for numerical methods that rely on well-conditioned sets of points, such as polynomial interpolation.

    Args:
        points (List[float] or np.ndarray): A list or numpy array of complex points.

    Returns:
        np.ndarray: The points in Leja order.

    Raises:
        ValueError: If the input `points` is an empty list or array.
    """
    # Check if input is empty
    if len(points) == 0:
        raise ValueError("Input must be a non-empty list or array of points.")

    # Convert to numpy array for easier manipulation
    points = np.array(points)

    # Start with the point having the smallest magnitude
    idx = np.argmin(np.abs(points))
    leja_ordered_points = [points[idx]]
    # Remove this point from further consideration
    points = np.delete(points, idx)

    while len(points) > 0:
        # Compute distances from remaining points to the last point in leja_order
        distances = np.abs(points - leja_ordered_points[-1])

        # Find the index of the point with the maximum minimum distance
        next_idx = np.argmax(distances)

        # Append this point to the leja_ordered_points
        leja_ordered_points.append(points[next_idx])

        # Remove this point from further consideration
        points = np.delete(points, next_idx)

    return np.array(leja_ordered_points)


def seprts(p: List[float]) -> np.ndarray:
    """
    Separate the roots of a polynomial into those inside the unit circle, on the unit circle, and outside the unit circle.

    Args:
        p (List[float]): The coefficients of the polynomial.

    Returns:
        np.ndarray: The roots of the polynomial, with those on the unit circle sorted by angle and every other one selected.
    """
    SN = 0.0001
    rts = np.roots(p)
    irts = rts[np.abs(rts) < (1 - SN)]  # Roots inside the unit circle
    orts = rts[np.isclose(np.abs(rts), 1, atol=SN)]  # Roots on the unit circle

    N = len(orts)
    if N % 2 != 0:
        print("Sorry, but there is a problem (1) in seprts function")
        return []

    # Sort roots on the unit circle by angle
    angles = np.angle(orts)
    sorted_indices = np.argsort(angles)
    orts = orts[sorted_indices[::2]]  # Select every second one after sorting

    # Make final list of roots
    r = np.concatenate((irts, orts))
    return r


def sfact(p: List[float]):
    """
    Compute the Schur-Cohn factorization of a polynomial.

    Args:
        p (List[float]): The coefficients of the polynomial.

    Returns:
        Tuple[np.ndarray, List[float]]: The Schur-Cohn factorization of the polynomial, where the first element is the normalized Schur-Cohn factor and the second element is the Leja-ordered roots of the polynomial.
    """
    if len(p) == 1:
        return p, []

    r = seprts(p)
    r_leja_ordered = leja_order(r.tolist())  # Convert list for leja function
    h = np.poly(r_leja_ordered)

    if np.all(np.isreal(p)):
        h = h.real

    # Normalize
    h *= np.sqrt(max(p) / np.sum(np.abs(h) ** 2))
    return h, r_leja_ordered


# Example usage
points = np.array([1 + 1j, 2 - 2j, 0.5 + 0.5j, -1 + 0j, 3 + 3j])
leja_ordered = leja_order(points)
print(leja_ordered)
