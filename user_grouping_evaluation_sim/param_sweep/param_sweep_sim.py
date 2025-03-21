import os
import sys
from typing import List
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from user_grouping_evaluation import run_movie

def simulation_for_params(params):
    """
    Helper to run a simulation for a single parameter combination.
    
    Parameters:
        params: A tuple (num_users, num_groups, num_experts, num_runs, num_of_rounds, epsilon)
    
    Returns:
        A tuple of DataFrames (df_stats, df_model) from run_movie.
    """
    num_users, num_groups, num_experts, num_runs, num_of_rounds, epsilon = params
    print(f"Process starting task for num_users={num_users}, num_groups={num_groups}")
        
    df_stats, df_model, model_history, expert_point, user_points, chosen_subset = run_movie(
        num_users=num_users,
        num_experts=num_experts,
        num_runs=num_runs,
        epsilon=epsilon,
        num_of_rounds=num_of_rounds,
        num_groups=num_groups
    )

    df_stats['num_users'] = num_users
    df_stats['num_groups'] = num_groups
    df_model['num_users'] = num_users
    df_model['num_groups'] = num_groups
    
    print(f"Finished task for num_users={num_users}, num_groups={num_groups}")
    return df_stats, df_model

def parameterSweep(num_users_list: List[int], 
                   num_groups_list: List[int], 
                   num_experts: int = 1, 
                   num_runs: int = 3, 
                   num_of_rounds: int = 100, 
                   epsilon: float = 5):
    """
    Run a parameter sweep over given lists of number of users and groups.
    Saves aggregated CSVs and produces plots.
    """
    tasks = []
    for num_users in num_users_list:
        for num_groups in num_groups_list:
            tasks.append((num_users, num_groups, num_experts, num_runs, num_of_rounds, epsilon))
    
    all_stats = []
    all_model = []
    # Use ProcessPoolExecutor to run simulations in parallel
    with ProcessPoolExecutor() as executor:
        for df_stats, df_model in executor.map(simulation_for_params, tasks):
            all_stats.append(df_stats)
            all_model.append(df_model)
    
    df_combined_stats = pd.concat(all_stats, ignore_index=True)
    df_combined_model = pd.concat(all_model, ignore_index=True)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    stats_dir = os.path.join(script_dir, "param_sweep_stats")

    os.makedirs(stats_dir, exist_ok=True)

    df_combined_stats.to_csv(os.path.join(stats_dir, "combined_simulation_stats.csv"), index=False)
    df_combined_model.to_csv(os.path.join(stats_dir, "combined_model_stats.csv"), index=False)
    
    df_avg = df_combined_stats.groupby(['Grouping Function', 'Evaluation Function', 'num_users', 'num_groups']).agg({
        'Pearson Correlation': 'mean',
        'Convergence Accuracy (final distance)': 'mean'
    }).reset_index()
    df_avg.to_csv(os.path.join(stats_dir, "aggregated_stats.csv"), index=False)

def main():
    num_users_list = [10, 25, 50, 75, 100]
    num_groups_list = [2, 3, 4, 5]
    
    parameterSweep(
        num_users_list=num_users_list, 
        num_groups_list=num_groups_list, 
        num_experts=1, 
        num_runs=50, 
        num_of_rounds=100, 
        epsilon=5
    )

if __name__ == "__main__":
    main()

