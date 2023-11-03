import numpy as np

def weighted_centroid(points: np.ndarray,
                      weights: np.ndarray) -> np.ndarray:
    """Compute the weighted centroid of a set of points.

    Args:
        points: The points.
        weights: The weights.

    Returns:
        The weighted centroid of the points.
    """
    # Check if the length of points and weights are the same
    if len(points) != len(weights):
        raise ValueError("Points and weights arrays must have the same length")

    # Compute the sum of weighted coordinates
    total_weighted_coords = np.sum(points * weights[:, np.newaxis], axis=0)

    # Compute the total weight
    total_weight = np.sum(weights)

    # normalize weights to sum to be between 0 and 1
    if np.allclose(weights, weights[0]):
        weights = np.ones(len(weights)) / len(weights)
    else:
        weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))

    # If total weight is zero, return an error
    if total_weight == 0:
        raise ValueError("Total weight is zero, cannot compute centroid")

    # Compute the weighted centroid coordinates
    C = total_weighted_coords / total_weight
    
    return C