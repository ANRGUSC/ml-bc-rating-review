#value_func.py
import numpy as np

### TODO: versions of this that use mean or median
## max gets worst from the expert
def dot_product(experts, coalition: np.ndarray, aggregate_function) -> float:
        """Returns the value of the coalition."""
        return 0 if len(coalition) <= 0 else np.max(np.dot(experts, aggregate_function(coalition, axis = 0)))


def l2_norm(experts, coalition: np.ndarray, aggregate_function) -> float:
        """Returns the value of the coalition."""
        return 0 if len(coalition) <= 0 else np.max(np.linalg.norm(experts, aggregate_function(coalition, axis = 0)))