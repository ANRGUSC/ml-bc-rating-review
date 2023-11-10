from dataclasses import dataclass
import pathlib
from copy import deepcopy
from functools import partial
from itertools import combinations
from math import factorial
import random
from typing import Callable, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from matplotlib.animation import FuncAnimation
from helpers import weighted_centroid


thisdir = pathlib.Path(__file__).parent.absolute()

@dataclass
class ProblemInstance:
    init_point: np.ndarray
    target_point: np.ndarray
    user_points: List[np.ndarray]

    def copy(self):
        return deepcopy(self)

class SimulationRun:
    def __init__(self,
                 problem: ProblemInstance,
                 point_history: List[np.ndarray] = [],
                 user_scores_history: List[float] = []) -> None:
        self.problem = problem
        self.point_history = point_history
        self.user_scores_history = user_scores_history

    def data_dist(self) -> pd.DataFrame:
        rows = []
        for i, point in enumerate(self.point_history):
            rows.append([i, np.linalg.norm(point - self.problem.target_point)])
        return pd.DataFrame(rows, columns=['step', 'distance'])
    
    def user_scores(self) -> pd.DataFrame:
        rows = []
        for i, scores in enumerate(self.user_scores_history):
            for user, score in enumerate(scores):
                rows.append([i, user, score])
        return pd.DataFrame(rows, columns=['step', 'user', 'score'])

class Simulator:
    def __init__(self,
                 init_score: Callable[[ProblemInstance], np.ndarray] = None,
                 get_groups: Callable[[SimulationRun], List[List[int]]] = None,
                 get_next_points: Callable[[SimulationRun, List[List[int]]], List[np.ndarray]] = None,
                 best_point: Callable[[SimulationRun, List[np.ndarray]], np.ndarray] = None,
                 update_scores: Callable[[SimulationRun, List[List[int]], List[np.ndarray]], np.ndarray] = None) -> None:
        self._init_score = init_score or self._default_init_score
        self._get_groups = get_groups or self._default_get_groups
        self._get_next_points = get_next_points or self._default_get_next_points
        self._best_point = best_point or self._default_best_point
        self._update_scores = update_scores or self._default_update_scores

    def _default_init_score(self, problem: ProblemInstance) -> np.ndarray:
        return np.ones(len(problem.user_points))

    def _default_get_groups(self, sim_run: SimulationRun) -> List[List[int]]:
        users_shuffled = np.random.permutation(list(range(len(sim_run.problem.user_points))))

        user_group_1 = users_shuffled[:len(users_shuffled) // 2]
        user_group_2 = users_shuffled[len(users_shuffled) // 2:]

        return [user_group_1, user_group_2]

    def _default_get_next_points(self, sim_run: SimulationRun, groups: List[List[int]]) -> int:
        centroids = [
            weighted_centroid(
                sim_run.problem.user_points[group],
                sim_run.user_scores_history[-1][group]
            )
            for group in groups
        ]

        current_point = sim_run.point_history[-1]
        delta = 0.1
        candidate_points = [
            current_point + delta * (centroid - current_point)
            for centroid in centroids
        ]
        return candidate_points

    def _default_best_point(self, sim_run: SimulationRun, next_points: List[np.ndarray]) -> np.ndarray:
        closest_point = None
        closest_distance = np.inf
        for point in next_points:
            distance = np.linalg.norm(point - sim_run.problem.target_point)
            if distance < closest_distance:
                closest_point = point
                closest_distance = distance
        
        current_point = sim_run.point_history[-1]
        current_distance = np.linalg.norm(current_point - sim_run.problem.target_point)
        if closest_distance > current_distance:
            return None
        return closest_point

    def _default_update_scores(self, sim_run: SimulationRun, groups: List[List[int]], next_points: List[np.ndarray]) -> np.ndarray:
        # reward groups with next_points == sim_run.point_history[-1]
        # return sim_run.user_scores_history[-1]
        winning_groups = [
            group
            for group, next_point in zip(groups, next_points)
            if np.allclose(next_point, sim_run.point_history[-1])
        ]

        new_scores = np.zeros(len(sim_run.problem.user_points))
        for group in winning_groups:
            new_scores[group] = 1

        return sim_run.user_scores_history[-1] + new_scores

    def run(self, problem: ProblemInstance, n_steps: int) -> SimulationRun:
        problem = problem.copy()

        user_scores = self._init_score(problem)
        sim_run = SimulationRun(
            problem=problem,
            point_history=[problem.init_point],
            user_scores_history=[user_scores]
        )

        for _ in range(n_steps):
            groups = self._get_groups(sim_run)
            next_points = self._get_next_points(sim_run, groups)
            next_point = self._best_point(sim_run, next_points)
            if next_point is not None:
                sim_run.point_history.append(next_point)
            else:
                sim_run.point_history.append(sim_run.point_history[-1])

            user_scores = self._update_scores(sim_run, groups, next_points)
            sim_run.user_scores_history.append(user_scores)

        return sim_run
    

def main():
    from graphing import init_points_movie, init_points_uniform, animate, get_shapley_values

    # Grouping Functions
    def exploit_vs_explore_rand(sim_run: SimulationRun,
                                p: float) -> List[List[int]]:
        if random.random() < p:
            users_shuffled = np.argsort(sim_run.user_scores_history[-1])
        else:
            users_shuffled = np.random.permutation(list(range(len(sim_run.problem.user_points))))

        user_group_1 = users_shuffled[:len(users_shuffled) // 2]
        user_group_2 = users_shuffled[len(users_shuffled) // 2:]

        return [user_group_1, user_group_2]
    
    def exploit_vs_explore(sim_run: SimulationRun,
                           num_explore_rounds: int) -> List[List[int]]:
        if len(sim_run.point_history) < num_explore_rounds:
            users_shuffled = np.random.permutation(list(range(len(sim_run.problem.user_points))))
        else:
            users_shuffled = np.argsort(sim_run.user_scores_history[-1])

        user_group_1 = users_shuffled[:len(users_shuffled) // 2]
        user_group_2 = users_shuffled[len(users_shuffled) // 2:]

        return [user_group_1, user_group_2]

    group_funcs = {
        **{f'rand_{p:0.2f}': partial(exploit_vs_explore_rand, p=p) for p in np.linspace(0, 1, 11)},
        **{f'explore_{n}': partial(exploit_vs_explore, num_explore_rounds=n) for n in range(10, 51, 10)}
    }

    # Get Next Point Functions
    def weighted_next_points(sim_run: SimulationRun,
                             groups: List[List[int]],
                             weight: Callable[[np.ndarray], np.ndarray]) -> int:
        centroids = [
            weighted_centroid(
                sim_run.problem.user_points[group],
                weight(sim_run.user_scores_history[-1][group])
            )
            for group in groups
        ]

        current_point = sim_run.point_history[-1]
        delta = 0.1
        candidate_points = [
            current_point + delta * (centroid - current_point)
            for centroid in centroids
        ]
        return candidate_points
    
    def norm_sig(x: np.ndarray, a: float = 10.0) -> np.ndarray:
        # normalize x to be between 0 and 1
        if np.allclose(x, x[0]):
            x = np.ones(len(x)) / len(x)
        else:
            x = (x - np.min(x)) / (np.max(x) - np.min(x))
        weights = 1 / (1 + np.exp(a*(-x + 1/2)))
        return weights

    next_points_funcs = {
        'score_sigmoid_1': partial(weighted_next_points, weight=partial(norm_sig, a=1.0)),
        'score_sigmoid_5': partial(weighted_next_points, weight=partial(norm_sig, a=5.0)),
        'score_sigmoid_10': partial(weighted_next_points, weight=partial(norm_sig, a=10.0)),
        'score_sigmoid_25': partial(weighted_next_points, weight=partial(norm_sig, a=25.0)),
        'score_sigmoid_50': partial(weighted_next_points, weight=partial(norm_sig, a=50.0)),
        # 'score': partial(weighted_next_points, weight=lambda x: x),
        # 'score_squared': partial(weighted_next_points, weight=lambda x: x**2),
    }

    modes = {
        'est': {},
        'no-est': {
            'update_scores': lambda sim_run, groups, next_points: sim_run.user_scores_history[-1]
        },
        # 'real': {
        #     'init_score': lambda problem: get_shapley_values(problem.user_points, problem.target_point),
        #     'update_scores': lambda sim_run, groups, next_points: sim_run.user_scores_history[-1]
        # }
    }

    num_runs = 20
    rows = []
    for group_func_name, group_func in group_funcs.items():
        for next_points_func_name, next_points_func in next_points_funcs.items():
            model_point, expert_point, user_points = init_points_movie(expert_point_subset_size=3, num_user_points=None)
            problem = ProblemInstance(model_point, expert_point, user_points)
            for mode, mode_kwargs in modes.items():
                for run_i in range(1, num_runs+1):
                    print(f"{group_func_name}, {next_points_func_name}, {mode} - {run_i}/{num_runs}", end='\n\r')
                    simulator = Simulator(
                        get_groups=group_func,
                        get_next_points=next_points_func,
                        **mode_kwargs
                    )
                    sim_run = simulator.run(problem, 100)
                    for i, point in enumerate(sim_run.point_history):
                        rows.append({
                            'group_func': group_func_name,
                            'next_points_func': next_points_func_name,
                            'mode': mode,
                            'run': run_i,
                            'step': i,
                            'distance': np.linalg.norm(point - problem.target_point)
                        })

                    df = pd.DataFrame(rows)
                    df.to_csv(thisdir / 'sim.csv', index=False)


if __name__ == '__main__':
    main()