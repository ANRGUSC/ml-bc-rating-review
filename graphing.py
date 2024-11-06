import pathlib
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
					   expert_point: np.ndarray) -> np.ndarray:
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
			subset_value = - \
				np.linalg.norm(
					np.mean(user_points[list(subset)], axis=0) - expert_point)
			# compute value of subset with user
			subset_with_user_value = - \
				np.linalg.norm(
					np.mean(user_points[list(subset) + [user]], axis=0) - expert_point)
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
		shapley_values = get_shapley_values(user_points, expert_point)
	else:
		shapley_values = np.ones(len(user_points))

	shapley_value_history: list = [deepcopy(shapley_values)]
	for round_i in range(num_of_rounds):
		# split user points into two random groups (use permutation)
		if random.random() < 0.1:
			# sort users by their shapley values
			users_shuffled = np.argsort(shapley_values)
		else:
			users_shuffled = np.random.permutation(
				list(range(len(user_points))))

		user_group_1 = users_shuffled[:len(users_shuffled) // 2]
		user_group_2 = users_shuffled[len(users_shuffled) // 2:]

		centroid_1 = weighted_centroid(
			user_points[user_group_1], shapley_values[user_group_1])
		centroid_2 = weighted_centroid(
			user_points[user_group_2], shapley_values[user_group_2])

		# create two candidate points, moving towards each centroid (use numpy) (by DELTA % of the distance)
		candidate_point_1 = model_point + delta * (centroid_1 - model_point)
		candidate_point_2 = model_point + delta * (centroid_2 - model_point)

		# let new model point be the one closer to the expert point
		dist_1 = np.linalg.norm(candidate_point_1 - expert_point)
		dist_2 = np.linalg.norm(candidate_point_2 - expert_point)
		dist_cur = np.linalg.norm(model_point - expert_point)

		if dist_cur < dist_1 and dist_cur < dist_2:
			# both points are further from expert point than current model point
			# don't update model point
			pass
		elif dist_1 < dist_2:
			model_point = candidate_point_1
			if do_estimation and not use_real_shapley:  # give user group 1 a point
				shapley_values[user_group_1] += 1
		else:
			model_point = candidate_point_2
			if do_estimation and not use_real_shapley:  # give user group 2 a point
				shapley_values[user_group_2] += 1

		model_point_history.append(model_point)
		shapley_value_history.append(deepcopy(shapley_values))

	return model_point_history, shapley_value_history

def scale_shapley_values(shapley_round, minval, maxval):
    """ Scale Shapley values linearly based on provided min and max values. """
    if np.max(shapley_round) == np.min(shapley_round):
        return shapley_round
    normalized = (shapley_round - np.min(shapley_round)) / (np.max(shapley_round) - np.min(shapley_round))
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
	rows_shapley = []
	stats = []  # List to store stats for each run and mode

	for run_i in range(num_runs):
		init_model_point, init_expert_point, init_user_points = init_points_movie(
			num_user_points=num_users,
			expert_point_subset_size=num_experts
		)

		starting_distance = np.linalg.norm(init_model_point - init_expert_point)

		model_point = deepcopy(init_model_point)
		expert_point = deepcopy(init_expert_point)
		user_points = deepcopy(init_user_points)

		model_point_history, shapley_history = simulation(model_point, expert_point, user_points, num_of_rounds=num_of_rounds)

		distances = np.array([np.linalg.norm(point - expert_point) for point in model_point_history])

		# Convergence Accuracy: Final distance to expert point
		final_distance = distances[-1]

		# Convergence Speed: Number of rounds to reach within epsilon
		convergence_indices = np.where(distances <= epsilon)[0]
		rounds_to_converge = convergence_indices[0] if convergence_indices.size > 0 else None

		# Collect model distance data
		rows_model.extend([[run_i, i, distance] for i, distance in enumerate(distances)])

		shapley_values = get_shapley_values(user_points, expert_point)

		scaled_shapley_history = np.array([scale_shapley_values(round_vals, np.min(shapley_values), np.max(shapley_values)) for round_vals in shapley_history])
		# Compute Shapley Accuracy: Average Shapley error over all agents and rounds
		rows_shapley.extend([
			[run_i, i, agent_num, sv, rv, abs(sv - rv)]
			for i, round_shapley_values in enumerate(scaled_shapley_history)
			for agent_num, (sv, rv) in enumerate(zip(round_shapley_values, shapley_values))
		])

		average_shapley_error = np.mean([row[-1] for row in rows_shapley])

		# Compute Average Shapley Error at the Convergence Round
		if rounds_to_converge is not None:
			shapley_errors_at_convergence = [
				row[-1] for row in rows_shapley if row[1] == rounds_to_converge
			]
			average_shapley_error_at_convergence = np.mean(shapley_errors_at_convergence)
		else:
			average_shapley_error_at_convergence = None

		stats.append({
			'Run': run_i,
			'Starting Distance': starting_distance,
			'Convergence Speed (rounds)': rounds_to_converge,
			'Convergence Accuracy (final distance)': final_distance,
			'Average Shapley Error': average_shapley_error,
			'Average Shapley Error at Convergence': average_shapley_error_at_convergence
		})

	# Convert stats to DataFrame for display
	df_stats = pd.DataFrame(stats)
	average_stats = df_stats.drop(columns=['Run']).mean(numeric_only=True).to_frame().T
	print("\nAverage Simulation Stats Across all Runs:")
	print(average_stats.to_string(index=False))

	# Identify best, median, and worst runs based on Convergence Accuracy (final distance)
	df_stats_sorted = df_stats.sort_values(by="Convergence Accuracy (final distance)")
	best_run = df_stats_sorted.iloc[0]["Run"]
	median_run = df_stats_sorted.iloc[len(df_stats_sorted) // 2]["Run"]
	worst_run = df_stats_sorted.iloc[-1]["Run"]

	df_model = pd.DataFrame(rows_model, columns=["Run", "Round", "Distance"])
	df_shapley = pd.DataFrame(rows_shapley, columns=["Run", "Round", "Agent", "Shapley Value", "Real Shapley Value", "Error"])

	# Case when num_runs < 3: Save only the best run plots
	if num_runs < 3:
		last_run_model = df_model[df_model["Run"] == best_run]
		last_run_shapley = df_shapley[df_shapley["Run"] == best_run]

		# Distance to Expert Point plot for the best run
		fig_distance = px.line(last_run_model, x="Round", y="Distance", template="plotly_white",
							title="Distance to Expert Point over Time")
		fig_distance.write_image("distance_to_expert_point.png")

		# Average Shapley Error plot for the best run
		df_shapley_error_best = last_run_shapley.groupby("Round")["Error"].mean().reset_index()
		fig_shapley_error = px.line(df_shapley_error_best, x="Round", y="Error", template="plotly_white",
									title="Average Shapley Error Over Time")
		fig_shapley_error.write_image("average_shapley_error_over_time.png")

		# Shapley Value Error by Agent for the best run
		fig_shapley_error_agents = px.line(last_run_shapley, x="Round", y="Error", color="Agent", template="plotly_white",
										title="Shapley Value Error Over Time by Agent")
		fig_shapley_error_agents.write_image("shapley_error.png")

	# Case when num_runs >= 3: Save plots for best, median, and worst runs
	else:
		for run_type, run_num in [("best", best_run), ("median", median_run), ("worst", worst_run)]:
			run_model = df_model[df_model["Run"] == run_num]
			run_shapley = df_shapley[df_shapley["Run"] == run_num]

			# Distance to Expert Point plot
			fig_distance = px.line(run_model, x="Round", y="Distance", template="plotly_white",
								title=f"Distance to Expert Point over Time ({run_type.capitalize()} Run: {run_num})")
			fig_distance.write_image(f"{run_type}_distance_to_expert_point.png")

			# Average Shapley Error plot
			df_shapley_error_run = run_shapley.groupby("Round")["Error"].mean().reset_index()
			fig_shapley_error = px.line(df_shapley_error_run, x="Round", y="Error", template="plotly_white",
										title=f"Average Shapley Error Over Time ({run_type.capitalize()} Run: {run_num})")
			fig_shapley_error.write_image(f"{run_type}_average_shapley_error_over_time.png")

			# Shapley Value Error by Agent
			fig_shapley_error_agents = px.line(run_shapley, x="Round", y="Error", color="Agent", template="plotly_white",
											title=f"Shapley Value Error Over Time by Agent ({run_type.capitalize()} Run: {run_num})")
			fig_shapley_error_agents.write_image(f"{run_type}_shapley_error.png")

if __name__ == "__main__":
	# main()
	run_movie(num_users=10, num_experts=1, num_runs=50)
