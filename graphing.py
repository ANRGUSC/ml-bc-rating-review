from itertools import combinations
from math import factorial
from typing import Iterable, List, Tuple
import matplotlib.pyplot as plt
import random
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import numpy as np
import plotly.express as px
import pandas as pd

#TODO: Simulate it in multiple dimensions, track the distance to expert point over time (see if it converges)
#TODO: Use estimated shapley values to compute weighted centroid of each group

# Constants
NUM_USER_POINTS = 10
NUM_OF_ROUNDS = 100

# Initialization
def initialize_points(dimension):
    model_point = np.random.uniform(0, 1, size= dimension)
    expert_point = np.random.uniform(0, 1, size= dimension)
    user_points = np.random.uniform(0, 1, size=(NUM_USER_POINTS, dimension))

    centroid = np.mean(user_points, axis=0)
    max_distance = np.max(np.linalg.norm(user_points - centroid, axis=1))
    expert_point = centroid + np.random.uniform(0, max_distance / 3, size=dimension)
    
    return model_point, expert_point, user_points

#model_point, expert_point, user_points = initialize_points()


def update_model_point(model_point):
    """Update the position of the model_point and the round number."""
    # Randomly move the point within a range of [-0.5, 0.5] for both x and y coordinates
    new_x = model_point[0] + random.uniform(-0.5, 0.5)
    new_y = model_point[1] + random.uniform(-0.5, 0.5)
    # Increase the round number by 1
    new_round = model_point[2] + 1
    return [new_x, new_y, new_round]

# Visualization
def plot_all_points_movement(model_point_history, user_points, expert_point):
    """Plot the movement of the model point, user points, and expert point over rounds."""
    plt.figure(figsize=(10, 7))
    
    # Plot movement of model point
    xs = [point[0] for point in model_point_history]
    ys = [point[1] for point in model_point_history]
    rounds = [point[2] for point in model_point_history]
    plt.scatter(xs, ys, c=rounds, cmap='viridis', s=100, alpha=0.8, label="Model Point")
    plt.plot(xs, ys, '-o', alpha=0.6)
    
    # Plot user points
    user_xs = [point[0] for point in user_points]
    user_ys = [point[1] for point in user_points]
    plt.scatter(user_xs, user_ys, c='blue', s=100, alpha=0.6, label='User Points')
    
    # Plot expert point
    plt.scatter(*expert_point, c='green', s=150, alpha=0.8, marker='*', label='Expert Point')
    
    plt.colorbar().set_label('Round Number for Model Point')
    plt.title("Movement of Model Point and Position of User and Expert Points")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.grid(True)
    plt.show()

def animate_all_points_movement(model_point_histories, user_points, expert_point):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Movement of Model Point and Position of User and Expert Points")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.grid(True)

    # Plot user points and expert point (they remain static)
    user_xs = [point[0] for point in user_points]
    user_ys = [point[1] for point in user_points]
    ax.scatter(user_xs, user_ys, c='blue', s=100, alpha=0.6, label='User Points')
    ax.scatter(*expert_point, c='green', s=150, alpha=0.8, marker='*', label='Expert Point')

    # Model point's trace and current position
    all_xs = {}
    all_ys = {}
    lines = {}
    points = {}
    for i, model_point_history in enumerate(model_point_histories):
        all_xs[i] = [point[0] for point in model_point_history]
        all_ys[i] = [point[1] for point in model_point_history]
        line, = ax.plot([], [], '-o', color='purple', alpha=0.6)
        lines[i] = line
        point, = ax.plot([], [], 'o', alpha=0.8, markersize=10, label='Model Point')
        points[i] = point

    # Initialization function
    def init():
        nonlocal all_xs, all_ys, lines, points
        for line in lines.values():
            line.set_data([], [])
        for point in points.values():
            point.set_data([], [])
        return *lines.values(), *points.values()

    # Animation update function with modification to address the warning
    def update(frame):
        nonlocal all_xs, all_ys, lines, points
        for xs, ys, line, point in zip(all_xs.values(), all_ys.values(), lines.values(), points.values()):
            line.set_data(xs[:frame+1], ys[:frame+1])
            point.set_data([xs[frame]], [ys[frame]])  # Wrap values in a list
        return *lines.values(), *points.values()

    ani = FuncAnimation(fig, update, frames=NUM_OF_ROUNDS, init_func=init, blit=True, repeat=False)
    plt.show()


# # old shapely value function
#     # Compute actual Shapley values
#     all_users = list(range(num_users))
#     shapely_values = np.zeros(num_users)

#     for user in range(num_users):
#         # iterate over all subsets without user
#         for subset in all_subsets(all_users, exclude=[user]):
#             # compute value of subset
#             subset_value = value_func(experts, users[list(subset)], aggregate)
#             # subset_value = ValueFunction.dot_product(users[list(subset)])
#             # compute value of subset with user
#             subset_with_user_value = value_func(experts, users[list(subset) + [user]], aggregate)
#             # compute marginal contribution of user
#             marginal_contribution = subset_with_user_value - subset_value
#             # update user value
#             weight = factorial(len(subset)) * factorial(num_users - len(subset) - 1) / factorial(num_users)
#             shapely_values[user] += weight * marginal_contribution

def all_subsets(elements: Iterable, exclude: Iterable = []) -> Iterable:
    """Returns all subsets (of length > 0) of elements excluding those in exclude"""
    # yield empty set
    for i in range(1, len(elements) + 1):
        for subset in combinations(elements, i):
            if not any(x in subset for x in exclude):
                yield subset

def get_shapley_values(user_points, expert_point):
    """Calculate the Shapley values for each user point.
    
    The coalition value function is the 1/distance of the centroid of the coalition to the expert point.
    """
    num_users = len(user_points)
    all_users = list(range(num_users))
    shapley_values = np.zeros(num_users)

    for user in range(num_users):
        # iterate over all subsets without user
        for subset in all_subsets(all_users, exclude=[user]):
            # compute value of subset
            subset_value = 1 / np.linalg.norm(np.mean(user_points[list(subset)], axis=0) - expert_point)
            # compute value of subset with user
            subset_with_user_value = 1 / np.linalg.norm(np.mean(user_points[list(subset) + [user]], axis=0) - expert_point)
            # compute marginal contribution of user
            marginal_contribution = subset_with_user_value - subset_value
            # update user value
            weight = factorial(len(subset)) * factorial(num_users - len(subset) - 1) / factorial(num_users)
            shapley_values[user] += weight * marginal_contribution
    return shapley_values

def weighted_centroid(points, weights):
    # Check if the length of points and weights are the same
    if len(points) != len(weights):
        raise ValueError("Points and weights arrays must have the same length")

    # Compute the sum of weighted coordinates
    total_weighted_coords = np.sum(points * weights[:, np.newaxis], axis=0)

    # Compute the total weight
    total_weight = np.sum(weights)

    # If total weight is zero, return an error
    if total_weight == 0:
        raise ValueError("Total weight is zero, cannot compute centroid")

    # Compute the weighted centroid coordinates
    C = total_weighted_coords / total_weight

    return C

def simulation(model_point, expert_point, user_points, use_real_shapley: bool = False) -> Tuple[list, np.ndarray, np.ndarray, list]:
    # convert user_points to numpy array
    user_points = np.array(user_points)
    model_point_history = [model_point]
    if use_real_shapley:
        shapley_values = get_shapley_values(user_points, expert_point)
        # normalize shapley values between 0 and 1
        shapley_values = (shapley_values - np.min(shapley_values)) / (np.max(shapley_values) - np.min(shapley_values))
    else:
        # shapley values start all equal
        user_score = np.ones(len(user_points))
        shapley_values = np.ones(len(user_points)) / len(user_points)

    shapley_value_history: list = [shapley_values]
    for _ in range(NUM_OF_ROUNDS):
        # split user points into two random groups (use permutation)
        users_shuffled = np.random.permutation(list(range(len(user_points))))
        user_group_1 = users_shuffled[:len(users_shuffled) // 2]
        user_group_2 = users_shuffled[len(users_shuffled) // 2:]

        # find weighted centroid of each group using normalized shapley values
        ## sum of all "delta" vectors scaled by shapley values
        centroid_1 = weighted_centroid(user_points[user_group_1], shapley_values[user_group_1])
        centroid_2 = weighted_centroid(user_points[user_group_2], shapley_values[user_group_2])

        # if not use_real_shapley, update shapley values based on winning group
        # whichever groups centroid is closer to the expert point, update shapley values
        if not use_real_shapley:
            if np.linalg.norm(centroid_1 - expert_point) < np.linalg.norm(centroid_2 - expert_point):
                user_score[user_group_1] += 1
            else:
                user_score[user_group_2] += 1
            shapley_values = user_score / np.sum(user_score)
            shapley_value_history.append(shapley_values)

        # create two candidate points, moving towards each centroid (use numpy) (by 10% of the distance)
        d = 0.1
        candidate_point_1 = model_point + d * (centroid_1 - model_point)
        candidate_point_2 = model_point + d * (centroid_2 - model_point)

        # let new model point be the one closer to the expert point
        if np.linalg.norm(candidate_point_1 - expert_point) < np.linalg.norm(candidate_point_2 - expert_point):
            model_point = candidate_point_1
        else:
            model_point = candidate_point_2
        # model_point = update_model_point(model_point)
        model_point_history.append(model_point)
    
    return model_point_history, shapley_value_history

def test():
    model_point = np.array([1, 1])
    expert_point = np.array([5, 5])
    user_points = np.array([[1, 2], [2, 1], [2, 2], [3, 3], [4, 4]])
    shapley_values = get_shapley_values(user_points, expert_point)
    normalized_shapley_values = (shapley_values - np.min(shapley_values)) / (np.max(shapley_values) - np.min(shapley_values))
    print(shapley_values)

    group_1 = [0, 1, 2]
    group_2 = [3, 4]

    # compute weighted centroid of each group
    real_centroid_1 = np.mean(user_points[group_1], axis=0)
    real_centroid_2 = np.mean(user_points[group_2], axis=0)
    centroid_1 = weighted_centroid(user_points[group_1], normalized_shapley_values[group_1])
    centroid_2 = weighted_centroid(user_points[group_2], normalized_shapley_values[group_2])


    # plot each point, color by shapley value
    plt.figure(figsize=(10, 7))
    plt.scatter(user_points[:, 0], user_points[:, 1], c=shapley_values, cmap='viridis', s=100, alpha=0.8)

    # plot expert point as star
    plt.scatter(*expert_point, c='green', s=150, alpha=0.8, marker='*', label='Expert Point')

    # plot centroids as triangles
    plt.scatter(*centroid_1, c='red', s=150, alpha=0.8, marker='^', label='Centroid 1')
    plt.scatter(*centroid_2, c='blue', s=150, alpha=0.8, marker='^', label='Centroid 2')

    # plot real centroids as squares
    plt.scatter(*real_centroid_1, c='red', s=150, alpha=0.8, marker='s', label='Real Centroid 1')
    plt.scatter(*real_centroid_2, c='blue', s=150, alpha=0.8, marker='s', label='Real Centroid 2')

    plt.colorbar().set_label('Shapley Value')
    plt.title("Shapley Values for User Points")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid(True)
    plt.show()


def main():
    model_point, expert_point, user_points = initialize_points(2)

    model_point_history, _ = simulation(model_point, expert_point, user_points)
    model_point_history_real, _ = simulation(model_point, expert_point, user_points, use_real_shapley=True)

    animate_all_points_movement([model_point_history, model_point_history_real], user_points, expert_point)

def multiple_dimension(dims: List[int] = [2**i for i in range(1, 7)]):
    rows_model = []
    rows_shapley = []
    for dim in dims:
        model_point, expert_point, user_points = initialize_points(dim)
        shapley_values = get_shapley_values(user_points, expert_point)
        shapley_values = (shapley_values - np.min(shapley_values)) / (np.max(shapley_values) - np.min(shapley_values))
        model_point_history, shapley_history = simulation(model_point, expert_point, user_points, use_real_shapley=False)
        distances = np.array([
            np.linalg.norm(point - expert_point)
            for point in model_point_history
        ])
        for i, distance in enumerate(distances):
            rows_model.append([dim, i, distance])

        for i, round_shapley_values in enumerate(shapley_history):
            for agent_num, (shapley_value, real_shapley_value) in enumerate(zip(round_shapley_values, shapley_values)):
                rows_shapley.append([dim, i, agent_num, shapley_value, real_shapley_value])

    df_model = pd.DataFrame(rows_model, columns=["Dimensions", "Round", "Distance"])
    fig = px.line(
        df_model, x="Round", y="Distance", color="Dimensions",
        template="plotly_white", title="Distance to Expert Point over Time"
    )
    fig.show()
    fig.write_image("distance.png")

    df_shapley = pd.DataFrame(rows_shapley, columns=["Dimensions", "Round", "Agent", "Shapley Value", "Real Shapley Value"])
    df_shapley['error'] = df_shapley['Shapley Value'] - df_shapley['Real Shapley Value']
    fig = px.line(
        df_shapley, x="Round", y="error", color="Agent", facet_col="Dimensions", facet_col_wrap=3,
        template="plotly_white", title="Shapley Value Error over Time"
    )
    fig.show()
    fig.write_image("shapley_error.png")




if __name__ == "__main__":
    # main()
    # test()
    multiple_dimension()
    

    
