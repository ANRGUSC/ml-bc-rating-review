#group_func.py

import numpy as np

def permutation_split(num_entities: int, num_groups: int) -> np.ndarray:
    """Generates a permutation of the entities and splits them into the given number of groups."""
    permutation = np.random.permutation(num_entities)
    return np.array_split(permutation, num_groups)
    
