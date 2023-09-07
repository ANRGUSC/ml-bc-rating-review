#value_func.py

import numpy as np
from enum import Enum

class ValueFunction:
    def __init__(self, experts: np.ndarray, aggregate_function=None):
        self.experts = experts
        self.aggregate_function = aggregate_function if aggregate_function else np.mean

        #Differnent Aggregate functions needs to be discussed

    def dot_product(self, coalition: np.ndarray) -> float:
        """Returns the value of the coalition."""
        return 0 if len(coalition) <= 0 else np.max(np.dot(self.experts, self.aggregate_function(coalition, axis = 0)))
    
class ValueFunction_Methods(Enum):
    DOT_PRODUCT = 1
    
    @staticmethod
    def get_value_function(method, experts, aggregate_function=None):
        if method == ValueFunction_Methods.DOT_PRODUCT:
            return ValueFunction(experts, aggregate_function)
        else:
            raise Exception("Invalid value function method")