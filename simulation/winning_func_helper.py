#value_func.py
import numpy as np

def max_points(values) -> float:
        """Returns the value of the coalition."""
        return np.argmax(values)