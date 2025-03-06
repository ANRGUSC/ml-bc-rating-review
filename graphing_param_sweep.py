import os
from typing import List
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import plotly.express as px
from graphing import run_movie  # Import the individual-run function

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
    
    df_stats, df_model = run_movie(
        num_users=num_users,
        num_experts=num_experts,
        num_runs=num_runs,
        epsilon=epsilon,
        num_of_rounds=num_of_rounds,
        num_groups=num_groups
    )
    # Annotate results with current parameters
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
    
    # Save combined data for later analysis
    os.makedirs("param_sweep_stats", exist_ok=True)
    df_combined_stats.to_csv("param_sweep_stats/combined_simulation_stats.csv", index=False)
    df_combined_model.to_csv("param_sweep_stats/combined_model_stats.csv", index=False)
    
    # Aggregate results across runs (average the metrics for each parameter combination)
    df_avg = df_combined_stats.groupby(['num_users', 'num_groups']).agg({
        'Pearson Correlation': 'mean',
        'Convergence Accuracy (final distance)': 'mean'
    }).reset_index()
    df_avg.to_csv("param_sweep_stats/aggregated_stats.csv", index=False)
    
    # Create plots in a separate folder
    os.makedirs("param_sweep_plots", exist_ok=True)
    
    # Plot 1: Pearson Coefficient vs. Number of Users
    fig_pearson = px.line(
        df_avg,
        x='num_users',
        y='Pearson Correlation',
        color='num_groups',
        markers=True,
        labels={
            'num_users': 'Number of Users',
            'Pearson Correlation': 'Avg Pearson Coefficient',
            'num_groups': 'Number of Groups'
        },
        title='Pearson Coefficient vs. Number of Users'
    )
    fig_pearson.write_image("param_sweep_plots/pearson_vs_users.png")
    
    # Plot 2: Convergence Distance vs. Number of Users
    fig_distance = px.line(
        df_avg,
        x='num_users',
        y='Convergence Accuracy (final distance)',
        color='num_groups',
        markers=True,
        labels={
            'num_users': 'Number of Users',
            'Convergence Accuracy (final distance)': 'Avg Convergence Distance',
            'num_groups': 'Number of Groups'
        },
        title='Convergence Distance vs. Number of Users'
    )
    fig_distance.write_image("param_sweep_plots/distance_vs_users.png")
    
    # Plot 3: Heatmap for Pearson Coefficient
    pivot_pearson = df_avg.pivot(index='num_users', columns='num_groups', values='Pearson Correlation')
    fig_heat_pearson = px.imshow(
        pivot_pearson,
        labels={
            "x": "Number of Groups",
            "y": "Number of Users",
            "color": "Avg Pearson Coefficient"
        },
        title="Heatmap: Avg Pearson Coefficient",
        aspect="auto"
    )
    fig_heat_pearson.write_image("param_sweep_plots/heatmap_pearson.png")
    
    # Plot 4: Heatmap for Convergence Distance
    pivot_distance = df_avg.pivot(index='num_users', columns='num_groups', values='Convergence Accuracy (final distance)')
    fig_heat_distance = px.imshow(
        pivot_distance,
        labels={
            "x": "Number of Groups",
            "y": "Number of Users",
            "color": "Avg Convergence Distance"
        },
        title="Heatmap: Avg Convergence Distance",
        aspect="auto"
    )
    fig_heat_distance.write_image("param_sweep_plots/heatmap_distance.png")

def main():
    # Define the parameter ranges for the sweep
    num_users_list = [10, 20, 25, 50, 75, 100]
    num_groups_list = [2, 3, 4, 5]
    
    # Run the parameter sweep
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
