#value_func.py

import numpy as np
from enum import Enum
def dot_product(self, coalition: np.ndarray) -> float:
        """Returns the value of the coalition."""
        return 0 if len(coalition) <= 0 else np.max(np.dot(self.experts, self.aggregate_function(coalition, axis = 0)))