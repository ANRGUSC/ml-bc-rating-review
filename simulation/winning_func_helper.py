#value_func.py
import numpy as np
from typing import Dict, Hashable

def max_points(values) -> Dict[Hashable, float]:
        """Returns the value of the coalition."""
        return {np.argmax(values): 1}