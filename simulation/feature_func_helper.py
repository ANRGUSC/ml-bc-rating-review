# feature_gen.py

import numpy as np
from enum import Enum


def uniform(num_entities: int, num_features: int) -> np.ndarray:
    """Generates a feature matrix with uniform distribution for the given number of entities and features."""
    return np.random.rand(num_entities, num_features)

def gaussian(num_entities: int, num_features: int, mean: float = 0.0, std_dev: float = 1.0) -> np.ndarray:
    """Generates a feature matrix with Gaussian distribution for the given number of entities and features."""
    return mean + std_dev * np.random.randn(num_entities, num_features)

def triangular(num_entities: int, num_features: int, left: float = 0.0, mode: float = 0.5, right: float = 1.0) -> np.ndarray:
    """Generates a feature matrix with triangular distribution for the given number of entities and features."""
    return np.random.triangular(left, mode, right, (num_entities, num_features))

    