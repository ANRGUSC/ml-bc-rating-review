# value_func.py
import numpy as np

def dot_product_mean(experts, coalition, aggregate_function):
    if len(coalition) <= 0:
        return 0

    dot_product = np.dot(experts, aggregate_function(coalition, axis=0))
    return np.mean(dot_product)


def dot_product_median(experts, coalition, aggregate_function):
    if len(coalition) <= 0:
        return 0
    dot_product = np.dot(experts, aggregate_function(coalition, axis=0))
    return np.median(dot_product)

# max gets worst from the expert


def dot_product(experts, coalition: np.ndarray, aggregate_function) -> float:
    """Returns the value of the coalition."""
    # print(experts)
    return 0 if len(coalition) <= 0 else np.max(np.dot(experts, aggregate_function(coalition, axis=0)))


def l2_norm(experts, coalition: np.ndarray, aggregate_function) -> float:
    """Returns the value of the coalition."""
    return 0 if len(coalition) <= 0 else np.max(np.linalg.norm(experts - aggregate_function(coalition, axis=0)))


def l2_norm_median(experts, coalition, aggregate_function):
    return 0 if len(coalition) <= 0 else np.median(np.linalg.norm(experts - aggregate_function(coalition, axis=0)))


def l2_norm_mean(experts, coalition, aggregate_function):
    return 0 if len(coalition) <= 0 else np.mean(np.linalg.norm(experts - aggregate_function(coalition, axis=0)))

#TODO: Needs to be discussed
#dot product with gaussian noise
def dot_product_with_noise(experts, coalition, noise):
        if len(coalition) <= 0:
                return 0
        coalition_experts = experts[coalition]
        dot_product = np.dot(coalition_experts, np.ones(len(coalition)))
        return np.max(dot_product) + noise
