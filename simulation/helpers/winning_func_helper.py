#value_func.py
import numpy as np
from typing import Dict, Hashable

def max_points(values) -> Dict[Hashable, float]:
        """Returns the value of the coalition."""
        return {np.argmax(values): 1}

#This function awards the top 3 groups instead of one
#TODO：Needs discussion
def max_3_points(values) -> Dict[Hashable, float]:
        """Returns the value of the coalition."""
        return {np.argmax(values): 1, np.argpartition(values, -2)[-2]: 1, np.argpartition(values, -3)[-3]: 1}