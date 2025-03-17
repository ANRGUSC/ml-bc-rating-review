import pathlib
import os
from copy import deepcopy
from itertools import combinations
from math import factorial
import random
from typing import Iterable, Tuple
from scipy.stats import pearsonr
from PIL import ImageColor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import shap
from sklearn.decomposition import PCA

thisdir = pathlib.Path(__file__).parent.absolute()

BASE_DIR = os.path.join(os.getcwd(), "user_grouping_evaluation_sim")
os.makedirs(BASE_DIR, exist_ok=True)

def evaluation_l2_norm(point_a: np.ndarray, point_b: np.ndarray) -> float:
    return -np.linalg.norm(point_a - point_b)

def evaluation_dot_product(point_a: np.ndarray, point_b: np.ndarray) -> float:
    return np.dot(point_a, point_b)

def evaluation_l1_norm(point_a: np.ndarray, point_b: np.ndarray) -> float:
    return -np.sum(np.abs(point_a - point_b))

def grouping_random(user_points, point_values, model_point, round_i, num_of_rounds, num_groups=2):
    users_shuffled = np.random.permutation(len(user_points))
    return np.array_split(users_shuffled, num_groups)

def grouping_epsilon_greedy(user_points, point_values, model_point, round_i, num_of_rounds, num_groups=2):
    """Reduces randomness in grouping over time, favoring higher point-valued agents."""
    # Calculate the temperature parameter
    temperature = 1 - (round_i / num_of_rounds)  # Decreases from 1 to 0 over rounds
    probability = max(0.4, temperature)

    if random.random() < probability:
        # Random grouping
        users_shuffled = np.random.permutation(len(user_points))
    else:
        # Group by point values
        users_shuffled = np.argsort(point_values)

    return np.array_split(users_shuffled, num_groups)


def grouping_interleaved(user_points, point_values, model_point, round_i, num_of_rounds, num_groups=2):
    # Sort the indices by point_values
    sorted_indices = np.argsort(point_values)
    n = len(sorted_indices)
    mid = n // 2
    
    # Split into two halves: lower performers and upper performers.
    bottom_half = list(sorted_indices[:mid])
    top_half = list(sorted_indices[mid:])
    
    # Shuffle each half to add randomness.
    random.shuffle(top_half)
    random.shuffle(bottom_half)
    
    # Distribute the top half into groups using round-robin assignment.
    groups_top = [[] for _ in range(num_groups)]
    for i, idx in enumerate(top_half):
        groups_top[i % num_groups].append(idx)
    
    # Distribute the bottom half similarly.
    groups_bottom = [[] for _ in range(num_groups)]
    for i, idx in enumerate(bottom_half):
        groups_bottom[i % num_groups].append(idx)
    
    # Combine the corresponding top and bottom groups.
    groups = []
    for i in range(num_groups):
        groups.append(np.array(groups_top[i] + groups_bottom[i], dtype=int))
    
    return groups

evaluation_functions = {
    'l2_norm': evaluation_l2_norm,
    'dot_product': evaluation_dot_product,
    'l1_norm': evaluation_l1_norm
}

grouping_functions = {
    'random': grouping_random,
    'epsilon_greedy': grouping_epsilon_greedy,
    'interleaved': grouping_interleaved
}

# Generate separate plots for each combination of Grouping Function and Evaluation Function
def create_individual_shapley_plots(df_points_avg_last_round):
    shapley_plots_dir = os.path.join(BASE_DIR, 'shapley_plots')
    os.makedirs(shapley_plots_dir, exist_ok=True)

    unique_combinations = df_points_avg_last_round['Combined'].unique()

    for combination in unique_combinations:
        # Filter data for the current combination
        combination_data = df_points_avg_last_round[df_points_avg_last_round['Combined'] == combination]

        pearson_corr, _ = pearsonr(combination_data['Points'], combination_data['Real Shapley Value'])
        # Generate scatter plot
        shapley_fig = px.scatter(
            combination_data,
            x='Points',
            y='Real Shapley Value',
            color='Combined',  # This will still be unique since we're filtering by combination
            title=f'Shapley vs Points for {combination}',
            labels={
                'Points': 'Points',
                'Real Shapley Value': 'Shapley Value'
            },
            template='plotly_white'
        )

        shapley_fig.add_annotation(
            x=0.5,
            y=0.9,
            xref='paper',
            yref='paper',
            text=f"Pearson Correlation: {pearson_corr:.2f}",
            showarrow=False
        )

        # Save each plot as a separate PNG file
        filename = os.path.join('shapley_plots', f"shapley_{combination.replace(', ', '_').replace(' ', '_')}.pdf")
        shapley_fig.write_image(filename)

def create_distances_by_grouping(df_model_avg, output_dir):
    distance_plots_dir = os.path.join(BASE_DIR, 'distance_plots', output_dir)
    os.makedirs(distance_plots_dir, exist_ok=True)

    # Get unique grouping functions
    unique_grouping_functions = df_model_avg['Grouping Function'].unique()

    for grouping_function in unique_grouping_functions:
        # Filter data for the current grouping function
        data = df_model_avg[df_model_avg['Grouping Function'] == grouping_function]

        # Create the plot
        fig = px.line(
            data,
            x='Round',
            y='Distance',
            color='Evaluation Function',
            title=f"Average Distance to Expert Point Over Rounds ({grouping_function})",
            labels={'Round': 'Round', 'Distance': 'Average Distance'},
            template='plotly_white'
        )

        # Save the plot
        filename = os.path.join(distance_plots_dir, f"{grouping_function}_distances.pdf".replace(" ", "_"))
        fig.write_image(filename)

    combined_fig = px.line(
        df_model_avg, 
        x='Round', 
        y='Distance', 
        color='Evaluation Function', 
        line_dash='Grouping Function', 
        template='plotly_white', 
        title='Average Distance to Expert Point Over Rounds'
    )        
    combined_filename = os.path.join(distance_plots_dir, 'combined_distances.pdf')
    combined_fig.write_image(combined_filename)

def create_median_pearson_plot(df_stats, base_dir, output_dir):
    """
    Create and save median-based Pearson correlation vs distance plots.
    
    Args:
        df_stats: DataFrame containing simulation statistics
        BASE_DIR: Base directory for outputs
        output_dir: Output directory name
    
    Returns:
        plotly.graph_objects.Figure: Figure object for the median plot
    """
    pearson_plots_dir = os.path.join(base_dir, 'pearson_plots')
    os.makedirs(pearson_plots_dir, exist_ok=True)
        
    df_agg = df_stats.groupby('Combined').agg(
        median_inverted_pearson=('Inverted Pearson', 'median'),
        q1_inverted_pearson=('Inverted Pearson', lambda x: x.quantile(0.25)),
        q3_inverted_pearson=('Inverted Pearson', lambda x: x.quantile(0.75)),
        median_distance=('Convergence Accuracy (final distance)', 'median'),
        q1_distance=('Convergence Accuracy (final distance)', lambda x: x.quantile(0.25)),
        q3_distance=('Convergence Accuracy (final distance)', lambda x: x.quantile(0.75))
    ).reset_index()
    
    df_agg[['Grouping Function', 'Evaluation Function']] = df_agg['Combined'].str.split(', ', expand=True)
    
    df_agg['error_x_plus'] = df_agg['q3_distance'] - df_agg['median_distance']
    df_agg['error_x_minus'] = df_agg['median_distance'] - df_agg['q1_distance']
    df_agg['error_y_plus'] = df_agg['q3_inverted_pearson'] - df_agg['median_inverted_pearson']
    df_agg['error_y_minus'] = df_agg['median_inverted_pearson'] - df_agg['q1_inverted_pearson']
    
    median_fig_with_error = px.scatter(
        df_agg,
        x='median_distance',
        y='median_inverted_pearson',
        color='Evaluation Function',
        symbol='Grouping Function',
        title='Median Inverse Pearson Correlation vs. Final Distance to Expert Point',
        template='plotly_white',
        error_x="error_x_plus",
        error_x_minus="error_x_minus",
        error_y="error_y_plus",
        error_y_minus="error_y_minus",
        labels={
            "median_distance": "Final Distance to Expert Point",
            "median_inverted_pearson": "Inverse Pearson Correlation"
        }
    )
    
    # Adjust error bar colors
    for trace in median_fig_with_error.data:
        if hasattr(trace, 'marker') and hasattr(trace.marker, 'color'):
            marker_color = trace.marker.color
            r, g, b = ImageColor.getrgb(marker_color)
            error_color = f"rgba({r},{g},{b},0.2)"
            if trace.error_x:
                trace.error_x.color = error_color
            if trace.error_y:
                trace.error_y.color = error_color
    
    subdir_path = os.path.join(pearson_plots_dir, output_dir)
    os.makedirs(subdir_path, exist_ok=True)
    
    median_filename = os.path.join(subdir_path, 'median_pearson_vs_distance.pdf')
    median_fig_with_error.write_image(median_filename)
    
def create_mean_pearson_plot(df_stats, base_dir, output_dir):
    """
    Create and save mean-based Pearson correlation vs distance plots.
    
    Args:
        df_stats: DataFrame containing simulation statistics
        pearson_plots_dir: Directory for Pearson plots
        output_dir: Output directory name
    
    Returns:
        plotly.graph_objects.Figure: Figure object for the mean plot
    """
    pearson_plots_dir = os.path.join(base_dir, 'pearson_plots')
    os.makedirs(pearson_plots_dir, exist_ok=True)
    
    df_agg_mean = df_stats.groupby('Combined').agg(
        mean_inverted_pearson=('Inverted Pearson', 'mean'),
        std_inverted_pearson=('Inverted Pearson', 'std'),
        mean_distance=('Convergence Accuracy (final distance)', 'mean'),
        std_distance=('Convergence Accuracy (final distance)', 'std')
    ).reset_index()
    
    df_agg_mean[['Grouping Function', 'Evaluation Function']] = df_agg_mean['Combined'].str.split(', ', expand=True)
    
    mean_percentile_fig = px.scatter(
        df_agg_mean,
        x='mean_distance',
        y='mean_inverted_pearson',
        color='Evaluation Function',
        symbol='Grouping Function',
        title='Average Inverse Pearson Correlation vs. Final Distance to Expert Point',
        template='plotly_white',
        error_x="std_distance",
        error_y="std_inverted_pearson",
        labels={
            "mean_distance": "Final Distance to Expert Point",
            "mean_inverted_pearson": "Inverse Pearson Correlation"
        }
    )
    
    # Adjust error bar colors
    for trace in mean_percentile_fig.data:
        if hasattr(trace, 'marker') and hasattr(trace.marker, 'color'):
            marker_color = trace.marker.color
            r, g, b = ImageColor.getrgb(marker_color)
            error_color = f"rgba({r},{g},{b},0.2)"
            if trace.error_x:
                trace.error_x.color = error_color
            if trace.error_y:
                trace.error_y.color = error_color
    
    subdir_path = os.path.join(pearson_plots_dir, output_dir)
    os.makedirs(subdir_path, exist_ok=True)
    
    mean_percentile_filename = os.path.join(subdir_path, 'average_pearson_vs_distance.pdf')
    mean_percentile_fig.write_image(mean_percentile_filename)   

def generate_sim_stats(stats, rows_model, output_dir):
    """
    Generate simulation statistics, save CSV files, and return processed DataFrames along with the output directory.
    
    Args:
        stats: List of simulation statistics dictionaries.
        rows_model: List of model distance data.
        num_users: Number of user points.
        num_groups: Number of groups.
        
    Returns:
        A tuple (df_stats, df_model, df_model_avg).
    """
    graphing_stats_dir = os.path.join(BASE_DIR, 'graphing_stats')
    os.makedirs(graphing_stats_dir, exist_ok=True)
    
    df_stats = pd.DataFrame(stats)
    df_stats['Inverted Pearson'] = df_stats['Pearson Correlation'].replace(0, np.nan).rdiv(1)

    sim_stat_dir = os.path.join(graphing_stats_dir, output_dir)
    os.makedirs(sim_stat_dir, exist_ok=True)
    df_stats.to_csv(os.path.join(sim_stat_dir, 'simulation_stats.csv'), index=False)
    
    df_stats['Combined'] = df_stats['Grouping Function'] + ', ' + df_stats['Evaluation Function']
    average_stats = df_stats.groupby(['Grouping Function', 'Evaluation Function']).mean(numeric_only=True).reset_index()
    average_stats = average_stats.drop(columns=['Run'])
    
    avg_stat_dir = os.path.join(graphing_stats_dir, output_dir)
    os.makedirs(avg_stat_dir, exist_ok=True)
    average_stats.to_csv(os.path.join(avg_stat_dir, 'average_simulation_stats.csv'), index=False)
    
    df_model = pd.DataFrame(rows_model, columns=["Grouping Function", "Evaluation Function", "Run", "Round", "Distance"])
    df_model_avg = df_model.groupby(['Grouping Function', 'Evaluation Function', 'Round']).mean().reset_index()
    df_model_avg['Combined'] = df_model_avg['Grouping Function'] + ', ' + df_model_avg['Evaluation Function']
    
    return df_stats, df_model, df_model_avg  
        
def init_points_movie(expert_point_subset_size: int,
                      num_user_points: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Initialize the model point, expert point, and user points.

    Args:
        expert_point_subset_size: The number of user points to use to calculate the expert point.
        num_user_points: The number of user points to use.

    Returns:
        A tuple of the model point, expert point, and user points.
    """
    
    path = thisdir / "movielens_data/user_genre_pivot_mean.csv"
    df_pref = pd.read_csv(path, encoding="utf-8", index_col=0)

    total = num_user_points + expert_point_subset_size

    df_pref = df_pref.iloc[:total]

    # Randomly sample users
    user_points = df_pref.sample(n=total, random_state=42).values

    # initial model point has same dimension but is drawn from uniform distribution
    model_point = np.random.uniform(0, 5, size=df_pref.shape[1])

    users_shuffled = np.random.permutation(user_points.shape[0])
    user_group = users_shuffled[:expert_point_subset_size]
    # expert point is centroid of random subset of user points
    expert_point = np.mean(user_points[user_group], axis=0)
    
    # remove expert_point from user_points
    user_points = np.delete(user_points, user_group, axis=0)

    return model_point, expert_point, user_points

def all_subsets(elements: Iterable, exclude: Iterable = []) -> Iterable:
    """Returns all subsets (of length > 0) of elements excluding those in exclude

    Args:
        elements: The elements to get subsets of.
        exclude: The elements to exclude.

    Returns:
        A generator of all subsets of elements excluding those in exclude.
    """
    for i in range(1, len(elements) + 1):
        for subset in combinations(elements, i):
            if not any(x in subset for x in exclude):
                yield subset

def kernel_shap_values(user_points, expert_point, evaluation_func, nsamples=10000):
    """KernelSHAP implementation for coalition value analysis."""
    num_features = len(user_points)

    # Create background data (a single reference point, representing an "empty" coalition)
    background = np.zeros((1, num_features))
    
    # Define model function for SHAP: it converts a binary mask to a coalition and evaluates it.
    def model_fn(mask_matrix):
        values = []
        for mask in mask_matrix:
            coalition = np.where(mask == 1)[0]
            if len(coalition) == 0:
                aggregated = np.zeros_like(expert_point)
            else:
                aggregated = np.mean(user_points[coalition], axis=0)
            values.append(evaluation_func(aggregated, expert_point))
        return np.array(values)
    
    # Initialize KernelExplainer with the model function and background data.
    explainer = shap.KernelExplainer(model_fn, background)
    
    l1_reg_string = f"num_features({num_features})"

    # Calculate SHAP values for full coalition (all users present).
    shap_values = explainer.shap_values(
        np.ones((1, len(user_points))),  # All users present
        nsamples=nsamples,
        l1_reg=l1_reg_string # Regularization to encourage sparsity
    )
    
    return shap_values[0]

def get_shapley_values(user_points: np.ndarray,
                       expert_point: np.ndarray,
                       evaluation_func) -> np.ndarray:
    """Calculate the Shapley values for each user point.

    The coalition value function is the 1/distance of the centroid of the coalition to the expert point.

    Args:
        user_points: The user points.
        expert_point: The expert point.

    Returns:
        The Shapley values for each user point.
    """
    num_users = len(user_points)
    all_users = list(range(num_users))
    shapley_values = np.zeros(num_users)

    for user in range(num_users):
        # iterate over all subsets without user
        for subset in all_subsets(all_users, exclude=[user]):
            # compute value of subset
            aggregated_subset = np.mean(user_points[list(subset)], axis=0)
            subset_value = evaluation_func(aggregated_subset, expert_point)

            # compute value of subset with user
            aggregated_subset_with_user = np.mean(user_points[list(subset) + [user]], axis=0)
            subset_with_user_value = evaluation_func(aggregated_subset_with_user, expert_point)
            
            # compute marginal contribution of user
            marginal_contribution = subset_with_user_value - subset_value
            
            # update user value
            weight = factorial(len(subset)) * factorial(num_users -
                                                        len(subset) - 1) / factorial(num_users)
            shapley_values[user] += weight * marginal_contribution
    return shapley_values

def weighted_centroid(points: np.ndarray,
                      weights: np.ndarray) -> np.ndarray:
    """Compute the weighted centroid of a set of points.

    Args:
        points: The points.
        weights: The weights.

    Returns:
        The weighted centroid of the points.
    """
    # Check if the length of points and weights are the same
    if len(points) != len(weights):
        raise ValueError("Points and weights arrays must have the same length")

    weights = weights**2

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

def simulation(model_point: np.ndarray,
               expert_point: np.ndarray,
               user_points: np.ndarray,
               evaluation_func,
               grouping_func,
               num_of_rounds: int = 100,
               delta: float = 0.1,
               error_probability: float = 0.05,
               num_groups: int = 2) -> Tuple[list, list, list]:
    """Simulate the movement of the model point over rounds.

    Args:
        model_point: The initial model point.
        expert_point: The expert point.
        user_points: The user points.
        evaluation_func: Function to evaluate model points.
        grouping_func: Function to group users.
        num_of_rounds: Number of rounds to simulate.
        delta: Step size for updates.
        error_probability: Probability of choosing a non-optimal candidate.
        num_groups: Number of groups to split users into.

    Returns:
        A tuple of (model_point_history, points_value_history, chosen_subsets).
    """

    user_points = np.array(user_points)
    model_point_history = [model_point]
    point_values = np.ones(len(user_points))
    points_value_history: list = [deepcopy(point_values)]
    chosen_subsets = []

    for round_i in range(num_of_rounds):
        groups = grouping_func(
            user_points, point_values, model_point, round_i, num_of_rounds, num_groups=num_groups)
        
        candidate_points = []
        for group in groups:
            centroid = weighted_centroid(user_points[group], point_values[group])
            candidate = model_point + delta * (centroid - model_point)
            candidate_points.append(candidate)

        value_cur = evaluation_func(model_point, expert_point)
        candidate_values = [evaluation_func(c, expert_point) for c in candidate_points]

        best_candidate_index = np.argmax(candidate_values)
        best_candidate_value = candidate_values[best_candidate_index]

        # Only update if any candidate improves on the current model
        if value_cur < best_candidate_value:
            # Choose candidate index with error probability (optional)
            if np.random.rand() < error_probability:
                available_indices = list(range(len(candidate_points)))
                available_indices.remove(best_candidate_index)
                chosen_index = random.choice(available_indices)
            else:
                chosen_index = best_candidate_index

            # For each group, update the group’s points by the marginal difference 
            # (which can be negative or positive)
            for i, group in enumerate(groups):
                reward = candidate_values[i] - value_cur
                point_values[group] += reward

            # Finally, update the model point using the candidate of the chosen group
            chosen_subsets.append(groups[chosen_index])
            model_point = candidate_points[chosen_index]
        else:
            chosen_subsets.append([])

        model_point_history.append(model_point)
        points_value_history.append(deepcopy(point_values))

    return model_point_history, points_value_history, chosen_subsets

def run_movie(num_users: int,
              num_experts: int,
              num_runs: int = 50,
              epsilon: float = 5,
              num_of_rounds: int = 100,
              num_groups: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame, list, np.ndarray, np.ndarray, list]:
    rows_model = []
    stats = []

    for grouping_name, grouping_func in grouping_functions.items():
        for eval_func_name, selected_eval_func in evaluation_functions.items():
            for run_i in range(num_runs):
                init_model_point, init_expert_point, init_user_points = init_points_movie(
                    num_user_points=num_users,
                    expert_point_subset_size=num_experts
                )

                starting_distance = np.linalg.norm(init_model_point - init_expert_point)

                model_point = deepcopy(init_model_point)
                expert_point = deepcopy(init_expert_point)
                user_points = deepcopy(init_user_points)

                model_history, points_history, chosen_subsets = simulation(
                    model_point,
                    expert_point, 
                    user_points, 
                    evaluation_func=selected_eval_func, 
                    grouping_func=grouping_func,
                    num_of_rounds=num_of_rounds,
                    num_groups=num_groups
                )

                distances = np.array([np.linalg.norm(point - expert_point) for point in model_history])

                # Convergence Accuracy: Final distance to expert point
                final_distance = distances[-1]

                # Convergence Speed: Number of rounds to reach within epsilon
                convergence_indices = np.where(distances <= epsilon)[0]
                rounds_to_converge = convergence_indices[0] if convergence_indices.size > 0 else None

                # Collect model distance data
                rows_model.extend([
                    [grouping_name, eval_func_name, run_i, i, distance] 
                    for i, distance in enumerate(distances)
                ])

                shapley_values = kernel_shap_values(user_points, expert_point, selected_eval_func)

                last_round_points = points_history[-1]

                # Compute Pearson Correlation between Shapley Values and last round of points
                if np.all(last_round_points == last_round_points[0]) or np.all(shapley_values == shapley_values[0]):
                    pearson_corr = np.nan
                else:
                    pearson_corr, _ = pearsonr(last_round_points, shapley_values)

                stats.append({
                    'Grouping Function': grouping_name,
                    'Evaluation Function': eval_func_name,
                    'Run': run_i,
                    'Starting Distance': starting_distance,
                    'Convergence Speed (rounds)': rounds_to_converge,
                    'Convergence Accuracy (final distance)': final_distance,
                    'Pearson Correlation': pearson_corr,
                    'Final Points': last_round_points,
                    'Shapley Values': shapley_values
                })

    output_dir = f"{num_users}_users_{num_groups}_groups"

    df_stats, df_model, df_model_avg = generate_sim_stats(stats, rows_model, output_dir)

    create_distances_by_grouping(df_model_avg, output_dir)
    
    create_mean_pearson_plot(df_stats, BASE_DIR, output_dir)
    create_median_pearson_plot(df_stats, BASE_DIR, output_dir)

    return df_stats, df_model, model_history, expert_point, user_points, chosen_subsets

if __name__ == "__main__":
    run_movie(num_users=75, num_experts=1, num_runs=50, num_groups=4)

