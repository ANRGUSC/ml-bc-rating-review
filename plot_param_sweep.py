#!/usr/bin/env python3
import os
import pandas as pd
import plotly.express as px

def main():
    # Read the combined simulation stats CSV
    csv_path = "./param_sweep_stats/combined_simulation_stats.csv"
    df = pd.read_csv(csv_path)
    
    # Ensure the output folder exists
    output_folder = "param_sweep_plots"
    os.makedirs(output_folder, exist_ok=True)
    
    # Aggregate the data by grouping function, evaluation function, num_users, and num_groups
    df_avg = df.groupby(
        ['Grouping Function', 'Evaluation Function', 'num_users', 'num_groups']
    ).agg({
        'Pearson Correlation': 'mean',
        'Convergence Accuracy (final distance)': 'mean'
    }).reset_index()
    
    df_avg.to_csv('param_sweep_stats/aggregated_stats.csv', index=False)
    
    # Get the unique combinations of grouping and evaluation methods
    unique_combos = df_avg[['Grouping Function', 'Evaluation Function']].drop_duplicates()

    for _, row in unique_combos.iterrows():
        grouping_func = row['Grouping Function']
        eval_func = row['Evaluation Function']
        
        # Filter data for this unique combination
        subset = df_avg[
            (df_avg['Grouping Function'] == grouping_func) &
            (df_avg['Evaluation Function'] == eval_func)
        ]
        
        # -----------------------------
        # Line Plot: Pearson vs. Users
        # -----------------------------
        fig_pearson_line = px.line(
            subset,
            x='num_users',
            y='Pearson Correlation',
            color='num_groups',
            markers=True,
            labels={
                'num_users': 'Number of Users',
                'Pearson Correlation': 'Avg Pearson Coefficient',
                'num_groups': 'Number of Groups'
            },
            title=f'Pearson Coefficient vs. Number of Users\n(Grouping: {grouping_func}, Evaluation: {eval_func})'
        )
        line_filename = os.path.join(
            output_folder,
            f"pearson_vs_users_{grouping_func.replace(' ', '_')}_{eval_func.replace(' ', '_')}.png"
        )
        fig_pearson_line.write_image(line_filename)
        
        # -------------------------------------
        # Line Plot: Convergence Distance vs. Users
        # -------------------------------------
        fig_distance_line = px.line(
            subset,
            x='num_users',
            y='Convergence Accuracy (final distance)',
            color='num_groups',
            markers=True,
            labels={
                'num_users': 'Number of Users',
                'Convergence Accuracy (final distance)': 'Avg Convergence Distance',
                'num_groups': 'Number of Groups'
            },
            title=f'Convergence Distance vs. Number of Users\n(Grouping: {grouping_func}, Evaluation: {eval_func})'
        )
        line_distance_filename = os.path.join(
            output_folder,
            f"distance_vs_users_{grouping_func.replace(' ', '_')}_{eval_func.replace(' ', '_')}.png"
        )
        fig_distance_line.write_image(line_distance_filename)
        
        # -----------------------------
        # Heatmap: Pearson Coefficient
        # -----------------------------
        pivot_pearson = subset.pivot(
            index='num_users', columns='num_groups', values='Pearson Correlation'
        )
        fig_heat_pearson = px.imshow(
            pivot_pearson,
            labels={
                "x": "Number of Groups",
                "y": "Number of Users",
                "color": "Avg Pearson Coefficient"
            },
            title=f"Heatmap: Avg Pearson Coefficient\n(Grouping: {grouping_func}, Evaluation: {eval_func})",
            aspect="auto"
        )
        heat_pearson_filename = os.path.join(
            output_folder,
            f"heatmap_pearson_{grouping_func.replace(' ', '_')}_{eval_func.replace(' ', '_')}.png"
        )
        fig_heat_pearson.write_image(heat_pearson_filename)
        
        # -----------------------------
        # Heatmap: Convergence Distance
        # -----------------------------
        pivot_distance = subset.pivot(
            index='num_users', columns='num_groups', values='Convergence Accuracy (final distance)'
        )
        fig_heat_distance = px.imshow(
            pivot_distance,
            labels={
                "x": "Number of Groups",
                "y": "Number of Users",
                "color": "Avg Convergence Distance"
            },
            title=f"Heatmap: Avg Convergence Distance\n(Grouping: {grouping_func}, Evaluation: {eval_func})",
            aspect="auto"
        )
        heat_distance_filename = os.path.join(
            output_folder,
            f"heatmap_distance_{grouping_func.replace(' ', '_')}_{eval_func.replace(' ', '_')}.png"
        )
        fig_heat_distance.write_image(heat_distance_filename)

if __name__ == "__main__":
    main()
