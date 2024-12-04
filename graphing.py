import pathlib
import os
from copy import deepcopy
from functools import partial
from itertools import combinations
from math import factorial
import random
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from matplotlib.animation import FuncAnimation

thisdir = pathlib.Path(__file__).parent.absolute()

# Evaluation Function Definitions
def evaluation_l2_norm(point_a: np.ndarray, point_b: np.ndarray) -> float:
	"""Evaluation function based on negative L2 norm (Euclidean distance)."""
	return -np.linalg.norm(point_a - point_b)

def evaluation_dot_product(point_a: np.ndarray, point_b: np.ndarray) -> float:
	"""Evaluation function based on dot product."""
	return np.dot(point_a, point_b)

def evaluation_cosine_similarity(point_a: np.ndarray, point_b: np.ndarray) -> float:
	"""Evaluation function based on cosine similarity with internal normalization."""
	# Normalize vectors
	norm_a = np.linalg.norm(point_a)
	norm_b = np.linalg.norm(point_b)
	if norm_a == 0 or norm_b == 0:
		return 0
	point_a_normalized = point_a / norm_a
	point_b_normalized = point_b / norm_b
	# Compute cosine similarity
	return np.dot(point_a_normalized, point_b_normalized)


def grouping_random(user_points, point_values, model_point, round_i, num_of_rounds):
	"""Hybrid grouping method: with 10% probability, group by Shapley values; else, random grouping."""
	users_shuffled = np.random.permutation(len(user_points))
	mid_point = len(users_shuffled) // 2
	user_group_1 = users_shuffled[:mid_point]
	user_group_2 = users_shuffled[mid_point:]
	return user_group_1, user_group_2


def grouping_simulated_annealing(user_points, point_values, model_point, round_i, num_of_rounds):
	"""Reduces randomness in grouping over time, favoring higher point-valued agents."""
	# Calculate the temperature parameter
	temperature = 1 - (round_i / num_of_rounds)  # Decreases from 1 to 0 over rounds
	probability = max(0.1, temperature)  # Ensure a minimum randomness of 10%

	if random.random() < probability:
		# Random grouping
		users_shuffled = np.random.permutation(len(user_points))
	else:
		# Group by point values
		users_shuffled = np.argsort(point_values)

	mid_point = len(users_shuffled) // 2
	return users_shuffled[:mid_point], users_shuffled[mid_point:]

def grouping_simulated_annealing_v2(user_points, point_values, model_point, round_i, num_of_rounds):
	"""
	Over time, assign more high-point users to user_group_1 while starting with random grouping.
	"""
	num_users = len(user_points)
	# Calculate the proportion of high-scoring users to assign to group_1
	progress_ratio = round_i / num_of_rounds  # Increases from 0 to 1 over the rounds
	num_high_scorers_to_assign = int(progress_ratio * num_users)  # Number of guaranteed high-point users

	# Get indices of users sorted by their point values (descending)
	sorted_indices = np.argsort(-point_values)  # Negative for descending order

	# Select the high scorers for group_1
	high_scorers = sorted_indices[:num_high_scorers_to_assign]

	# Randomly shuffle remaining users
	remaining_users = sorted_indices[num_high_scorers_to_assign:]
	np.random.shuffle(remaining_users)

	# Combine high scorers with the remaining users to form user_group_1 and user_group_2
	mid_point = len(remaining_users) // 2
	user_group_1 = np.concatenate([high_scorers, remaining_users[:mid_point]])
	user_group_2 = remaining_users[mid_point:]

	return user_group_1, user_group_2


def grouping_weighted_kmeans(user_points, point_values, model_point, round_i, num_of_rounds):
	"""Group users using K-means clustering with point_values as sample weights."""
	from sklearn.cluster import KMeans
	# Ensure point_values are non-negative
	point_values = np.maximum(point_values, 0)
	# Initialize KMeans with 2 clusters
	kmeans = KMeans(n_clusters=2, random_state=round_i)
	# Fit the model using sample weights
	kmeans.fit(user_points, sample_weight=point_values)
	labels = kmeans.labels_
	# Separate users into two groups based on labels
	group1 = np.where(labels == 0)[0]
	group2 = np.where(labels == 1)[0]
	return group1, group2


evaluation_functions = {
	'l2_norm': evaluation_l2_norm,
	'dot_product': evaluation_dot_product,
	'cosine_similarity': evaluation_cosine_similarity
}

grouping_functions = {
	'random': grouping_random,
	'simulated_annealing': grouping_simulated_annealing,
	'weighted_kmeans': grouping_weighted_kmeans,
	'simulated_annealing_v2': grouping_simulated_annealing_v2
}

def min_max_normalize(group):
	group['Points'] = (group['Points'] - group['Points'].min()) / (group['Points'].max() - group['Points'].min())
	group['Real Shapley Value'] = (group['Real Shapley Value'] - group['Real Shapley Value'].min()) / (group['Real Shapley Value'].max() - group['Real Shapley Value'].min())
	return group

# Generate separate plots for each combination of Grouping Function and Evaluation Function
def create_individual_shapley_plots(df_points_avg_last_round):
	os.makedirs('shapley_plots', exist_ok=True)

	unique_combinations = df_points_avg_last_round['Combined'].unique()

	for combination in unique_combinations:
		# Filter data for the current combination
		combination_data = df_points_avg_last_round[df_points_avg_last_round['Combined'] == combination]

		# Generate scatter plot
		shapley_fig = px.scatter(
		combination_data,
		x='Points',
		y='Real Shapley Value',
		color='Combined',  # This will still be unique since we're filtering by combination
		title=f'Shapley vs Points for {combination}',
		labels={
			'Points': 'Normalized Points',
			'Real Shapley Value': 'Normalized Real Shapley Value'
		},
		template='plotly_white'
		)

		# Save each plot as a separate PNG file
		filename = os.path.join('shapley_plots', f"shapley_{combination.replace(', ', '_').replace(' ', '_')}.png")
		shapley_fig.write_image(filename)

def create_distances_by_grouping(df_model_avg):
	os.makedirs('distance_plots', exist_ok=True)

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
		filename = os.path.join('distance_plots', f"{grouping_function}_distances.png".replace(" ", "_"))
		fig.write_image(filename)

# Initialization Functions
def init_points_uniform(dimension: int,
						num_user_points: int = 10,
						expert_point_subset_size: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Initialize the model point, expert point, and user points.

	Args:
		dimension: The dimension of the points.
		num_user_points: The number of user points.
		expert_point_subset_size: The number of user points to use to calculate the expert point.

	Returns:
		A tuple of the model point, expert point, and user points.
	"""
	model_point = np.random.uniform(0, 10, size=dimension)
	user_points = np.random.uniform(0, 10, size=(num_user_points, dimension))

	# expert point is centroid of random subset of user points
	users_shuffled = np.random.permutation(list(range(len(user_points))))
	user_group = users_shuffled[:expert_point_subset_size]
	expert_point = np.mean(user_points[user_group], axis=0)

	return model_point, expert_point, user_points


def init_points_movie(expert_point_subset_size: int,
					  num_user_points: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Initialize the model point, expert point, and user points.

	Args:
		expert_point_subset_size: The number of user points to use to calculate the expert point.
		num_user_points: The number of user points to use.

	Returns:
		A tuple of the model point, expert point, and user points.
	"""
	path = thisdir / "data/movielens/user_genre_pivot.csv"
	df_pref = pd.read_csv(path, encoding="utf-8", index_col=0)

	path = thisdir / "data/movielens/user_genre_pivot_std.csv"
	df_std = pd.read_csv(path, encoding="utf-8", index_col=0)

	path = thisdir / "data/movielens/user_genre_pivot_mean.csv"
	df_mean = pd.read_csv(path, encoding="utf-8", index_col=0)

	if df_pref.shape != df_std.shape or df_pref.shape != df_mean.shape:
		raise ValueError("DataFrames do not have matching dimensions")

	if num_user_points is not None:
		df_pref = df_pref.iloc[:num_user_points]
		df_mean = df_mean.iloc[:num_user_points]
		df_std = df_std.iloc[:num_user_points]

	user_points = np.zeros(df_pref.shape)
	for i in range(df_pref.shape[0]):
		for j in range(df_pref.shape[1]):
			user_points[i, j] = np.random.normal(
				df_mean.iloc[i, j], df_std.iloc[i, j])

	# initial model point has same dimension but is drawn from uniform distribution
	model_point = np.random.uniform(0, 5, size=df_pref.shape[1])

	users_shuffled = np.random.permutation(user_points.shape[0])
	user_group = users_shuffled[:expert_point_subset_size]
	# expert point is centroid of random subset of user points
	expert_point = np.mean(user_points[user_group], axis=0)

	return model_point, expert_point, user_points


def animate(model_point_histories: List[List[np.ndarray]],
			user_points: np.ndarray,
			expert_point: np.ndarray,
			colors: List[str] = None) -> None:
	"""Animate the movement of the model point over rounds.

	Args:
		model_point_histories: A list of model point histories.
		user_points: The user points.
		expert_point: The expert point.
		colors: The colors of the model points.
	"""
	fig: plt.Figure
	ax: plt.Axes
	fig, ax = plt.subplots(figsize=(10, 7))
	ax.set_xlim(0, 10)
	ax.set_ylim(0, 10)
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
	all_xs: Dict[int, List[float]] = {}
	all_ys: Dict[int, List[float]] = {}
	lines: Dict[int, plt.Line2D] = {}
	points: Dict[int, plt.Line2D] = {}
	for i, model_point_history in enumerate(model_point_histories):
		all_xs[i] = [point[0] for point in model_point_history]
		all_ys[i] = [point[1] for point in model_point_history]
		line, = ax.plot([], [], '-o', color='purple', alpha=0.6)
		lines[i] = line
		color = None if colors is None else colors[i]
		point, = ax.plot([], [], 'o', color=color, alpha=0.8, markersize=10, label='Model Point')
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

	num_rounds = len(model_point_histories[0])
	ani = FuncAnimation(fig, update, frames=num_rounds, init_func=init, blit=True, repeat=False)
	plt.show()


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

	# normalize weights to sum to be between 0 and 1
	if np.allclose(weights, weights[0]):
		weights = np.ones(len(weights)) / len(weights)
	else:
		weights = (weights - np.min(weights)) / \
			(np.max(weights) - np.min(weights))

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
			   use_real_shapley: bool = False,
			   do_estimation: bool = True) -> Tuple[list, np.ndarray, np.ndarray, list]:
	"""Simulate the movement of the model point over rounds.

	Args:
		model_point: The initial model point.
		expert_point: The expert point.
		user_points: The user points.
		use_real_shapley: Whether to use the real Shapley values or not.

	Returns:
		A tuple of the model point history, the Shapley value history, the centroid history, and the user point history.
	"""
	user_points = np.array(user_points)
	model_point_history = [model_point]
	if use_real_shapley:
		point_values = get_shapley_values(user_points, expert_point, evaluation_func)
	else:
		point_values = np.ones(len(user_points))

	points_value_history: list = [deepcopy(point_values)]
	for round_i in range(num_of_rounds):
		user_group_1, user_group_2 = grouping_func(
			user_points, point_values, model_point, round_i, num_of_rounds)

		centroid_1 = weighted_centroid(
			user_points[user_group_1], point_values[user_group_1])
		centroid_2 = weighted_centroid(
			user_points[user_group_2], point_values[user_group_2])

		# create two candidate points, moving towards each centroid (use numpy) (by DELTA % of the distance)
		candidate_point_1 = model_point + delta * (centroid_1 - model_point)
		candidate_point_2 = model_point + delta * (centroid_2 - model_point)

		# let new model point be the one closer to the expert point
		value_1 = evaluation_func(candidate_point_1, expert_point)
		value_2 = evaluation_func(candidate_point_2, expert_point)
		value_cur = evaluation_func(model_point, expert_point)

		if value_cur >= value_1 and value_cur >= value_2:
			# both points are further from expert point than current model point
			# don't update model point
			pass
		elif value_1 > value_2:
			model_point = candidate_point_1
			if do_estimation and not use_real_shapley:  # give user group 1 a point
				point_values[user_group_1] += 1
		else:
			model_point = candidate_point_2
			if do_estimation and not use_real_shapley:  # give user group 2 a point
				point_values[user_group_2] += 1

		model_point_history.append(model_point)
		points_value_history.append(deepcopy(point_values))

	return model_point_history, points_value_history

def scale_points_values(points_round, minval, maxval):
	""" Scale Point values linearly based on provided min and max values. """
	if np.max(points_round) == np.min(points_round):
		return points_round
	normalized = (points_round - np.min(points_round)) / (np.max(points_round) - np.min(points_round))
	return normalized * (maxval - minval) + minval

def main():
	model_point, expert_point, user_points = init_points_uniform(2)
	model_point_history, _ = simulation(
		model_point, expert_point, user_points, use_real_shapley=False)
	# model_point_history_real, shapley_value_history = simulation(
	#     model_point, expert_point, user_points, use_real_shapley=True)
	animate(
		[model_point_history],
		user_points, expert_point,
		colors=['purple', 'orange'],
	)


def run_movie(num_users: int,
			  num_experts: int,
			  num_runs: int = 1,
			  epsilon: float = 5,
			  num_of_rounds: int = 100):
	rows_model = []
	rows_points = []
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

				model_history, points_history = simulation(
					model_point,
					expert_point, 
					user_points, 
					evaluation_func=selected_eval_func, 
					grouping_func=grouping_func,
					num_of_rounds=num_of_rounds
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

				shapley_values = get_shapley_values(user_points, expert_point, selected_eval_func)

				scaled_points_history = np.array([scale_points_values(round_vals, np.min(shapley_values), np.max(shapley_values)) for round_vals in points_history])
				# Compute Shapley Accuracy: Average Shapley error over all agents and rounds
				rows_points.extend([
					[grouping_name, eval_func_name, run_i, i, agent_num, sv, rv]
					for i, round_point_values in enumerate(scaled_points_history)
					for agent_num, (sv, rv) in enumerate(zip(round_point_values, shapley_values))
				])

				stats.append({
					'Grouping Function': grouping_name,
					'Evaluation Function': eval_func_name,
					'Run': run_i,
					'Starting Distance': starting_distance,
					'Convergence Speed (rounds)': rounds_to_converge,
					'Convergence Accuracy (final distance)': final_distance
				})

	# Convert stats to DataFrame for display
	df_stats = pd.DataFrame(stats)
	average_stats = df_stats.groupby(['Grouping Function', 'Evaluation Function']).mean(numeric_only=True).reset_index()
	print("\nAverage Simulation Stats Across all Runs:")
	print(average_stats.to_string(index=False))

	df_model = pd.DataFrame(rows_model, columns=["Grouping Function", "Evaluation Function", "Run", "Round", "Distance"])
	df_model_avg = df_model.groupby(['Grouping Function', 'Evaluation Function', 'Round']).mean().reset_index()
	df_model_avg['Combined'] = df_model_avg['Grouping Function'] + ', ' + df_model_avg['Evaluation Function']
		
	create_distances_by_grouping(df_model_avg)

	# Save a .png called distances.png that shows average distance to expert point over all rounds with a line for each evaluation function
	distance_fig = px.line(df_model_avg, x='Round', y='Distance', color='Combined', template='plotly_white', title='Average Distance to Expert Point Over Rounds')
	filename = os.path.join('distance_plots', 'combined_distances.png')
	distance_fig.write_image(filename)

	df_points = pd.DataFrame(rows_points, columns=["Grouping Function", "Evaluation Function", "Run", "Round", "Agent", "Points", "Real Shapley Value"])
	df_points_avg = df_points.groupby(['Grouping Function', 'Evaluation Function', 'Agent', 'Round'])[['Points', 'Real Shapley Value']].mean().reset_index()

	# filter only the last round for each evaluation function
	df_points_avg_last_round = df_points_avg[df_points_avg['Round'] == num_of_rounds - 1]
	df_points_avg_last_round = df_points_avg_last_round.groupby(['Grouping Function', 'Evaluation Function']).apply(min_max_normalize)
	df_points_avg_last_round['Combined'] = df_points_avg_last_round['Grouping Function'] + ', ' + df_points_avg_last_round['Evaluation Function']

	create_individual_shapley_plots(df_points_avg_last_round)

	# Print the Shapley values DataFrame for debugging
	print(df_points_avg_last_round[df_points_avg_last_round['Evaluation Function'].isin(['dot_product', 'l2_norm', 'cosine_similarity'])].to_string(index=False))


if __name__ == "__main__":
	# main()
	run_movie(num_users=10, num_experts=1, num_runs=50)
