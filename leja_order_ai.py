import numpy as np
from typing import List


def leja_order(points):
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
