import numpy as np


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


# Example usage
points = np.array([1 + 1j, 2 - 2j, 0.5 + 0.5j, -1 + 0j, 3 + 3j])
leja_ordered = leja_order(points)
print(leja_ordered)
