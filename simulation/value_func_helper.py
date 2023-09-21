# value_func.py
import numpy as np

def dot_product_mean(experts, coalition):
    if len(coalition) <= 0:
        return 0
    coalition_experts = experts[coalition]
    dot_product = np.dot(coalition_experts, np.ones(len(coalition)))
    return np.mean(dot_product)


def dot_product_median(experts, coalition):
    if len(coalition) <= 0:
        return 0
    coalition_experts = experts[coalition]
    dot_product = np.dot(coalition_experts, np.ones(len(coalition)))
    return np.median(dot_product)

# max gets worst from the expert


def dot_product(experts, coalition: np.ndarray, aggregate_function) -> float:
    """Returns the value of the coalition."""
    return 0 if len(coalition) <= 0 else np.max(np.dot(experts, aggregate_function(coalition, axis=0)))


def l2_norm(experts, coalition: np.ndarray, aggregate_function) -> float:
    """Returns the value of the coalition."""
    return 0 if len(coalition) <= 0 else np.max(np.linalg.norm(experts, aggregate_function(coalition, axis=0)))


def l2_norm_median(experts, coalition):
    if len(coalition) <= 0:
        return 0
    coalition_experts = experts[coalition]
    l2_norm = np.linalg.norm(coalition_experts, ord=2, axis=1)
    return np.median(l2_norm)


def l2_norm_mean(experts, coalition):
    if len(coalition) <= 0:
        return 0
    coalition_experts = experts[coalition]
    l2_norm = np.linalg.norm(coalition_experts, ord=2, axis=1)
    return np.mean(l2_norm)

#TODO: Needs to be discussed
def dot_product_with_noise(experts, coalition, noise):
        if len(coalition) <= 0:
                return 0
        coalition_experts = experts[coalition]
        dot_product = np.dot(coalition_experts, np.ones(len(coalition)))
        return np.max(dot_product) + noise
