#value_func.py
import numpy as np

def dot_product(experts, coalition: np.ndarray, aggregate_function) -> float:
        """Returns the value of the coalition."""
        return 0 if len(coalition) <= 0 else np.max(np.dot(experts, aggregate_function(coalition, axis = 0)))