import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Extracted plotting function
def plot_subset_influence_static(
    user_points: np.ndarray,
    expert_point: np.ndarray,
    model_point_history: np.ndarray,
    chosen_subsets_per_round: list,
    round_indices: list = [0, 6, 12, 19],
    title: str = "How a Subset Pulls the Model Toward Its Centroid"
):
    """
    Creates a static multi-panel figure showing how, in specific rounds, the chosen subset
    pulls the model point toward its centroid.
    """
    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 16,
        'axes.labelsize': 16,
        'legend.fontsize': 17
    })
    
    # 1) Stack all points for PCA
    T = len(model_point_history)
    all_points = np.vstack([
        user_points,
        model_point_history,
        expert_point.reshape(1, -1)
    ])

    # 2) Reduce to 2D via PCA
    pca = PCA(n_components=2)
    all_points_2d = pca.fit_transform(all_points)

    # 3) Separate them back out
    N = len(user_points)
    user_points_2d = all_points_2d[:N]
    model_points_2d = all_points_2d[N : N + T]
    expert_point_2d = all_points_2d[N + T]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Calculate distances from model points to expert for color gradient
    distances = np.linalg.norm(model_points_2d - expert_point_2d, axis=1)
    max_dist = np.max(distances)
    min_dist = np.min(distances)
    
    for ax_idx, round_i in enumerate(round_indices):
        ax = axes[ax_idx]

        # Plot all user points in gray
        ax.scatter(
            user_points_2d[:, 0],
            user_points_2d[:, 1],
            color='gray',
            alpha=0.5,
            s=100,
            label='All Users' if ax_idx == 0 else ""
        )

        # Plot expert point in red
        ax.scatter(
            expert_point_2d[0],
            expert_point_2d[1],
            color='red',
            marker='X',
            s=250,
            label='Expert Point' if ax_idx == 0 else ""
        )

        # Plot model progression up to this round with increasingly darker purple
        for prev_round in range(min(round_i + 1, T)):
            model_2d = model_points_2d[prev_round]
            
            # Calculate color intensity based on distance to expert point
            dist = distances[prev_round]
            intensity = 0.3 + 0.7 * (1 - (dist - min_dist) / (max_dist - min_dist + 1e-10))
            
            if prev_round == round_i:
                edgecolor = 'black'
                linewidth = 2
                s = 180
                label = 'Current Model' if ax_idx == 0 else ""
            else:
                edgecolor = 'purple'
                linewidth = 1
                s = 160
                label = 'Model History' if prev_round == 0 and ax_idx == 0 else ""
            
            ax.scatter(
                model_2d[0],
                model_2d[1],
                facecolors=(128/255, 0, 128/255, intensity),
                edgecolors=edgecolor,
                linewidth=linewidth,
                s=s,
                label=label
            )

        # Highlight the chosen subset for this round
        subset_indices = chosen_subsets_per_round[round_i]
        subset_2d = user_points_2d[subset_indices]
        ax.scatter(
            subset_2d[:, 0],
            subset_2d[:, 1],
            color='orange',
            label='Chosen Subset' if ax_idx == 0 else "",
            s=120
        )

        # Compute and plot the centroid of that subset
        centroid_2d = subset_2d.mean(axis=0)
        ax.scatter(
            centroid_2d[0],
            centroid_2d[1],
            s=400,
            facecolors='orange',
            edgecolors='orange',
            alpha=0.3,
            linewidth=2,
            label='Subset Centroid' if ax_idx == 0 else ""
        )

        # Draw arrow from the current model point toward the centroid with fixed length
        current_model = model_points_2d[round_i]
        direction = centroid_2d - current_model
        norm = np.linalg.norm(direction)
        if not np.isclose(norm, 0):
            fixed_length = 1.25
            unit_direction = direction / norm
            arrow_end = current_model + unit_direction * fixed_length
        else:
            arrow_end = centroid_2d

        ax.annotate(
            '',
            xy=arrow_end,
            xytext=current_model,
            arrowprops=dict(
                facecolor='purple',
                edgecolor='purple',
                linewidth=1.5,
                arrowstyle='->',
                mutation_scale=15
            )
        )

        ax.set_title(f"Round {round_i + 1}", fontsize=20)
        ax.tick_params(axis='both', labelbottom=False, labelleft=False)

    # Create a single legend for the entire figure
    handles, labels = axes[0].get_legend_handles_labels()
    figure = fig.legend(handles, labels, loc='lower left', bbox_to_anchor=(0.0, 0.1))
    figure.get_frame().set_alpha(1)

    fig.suptitle(title, fontsize=26)
    plt.tight_layout()

    # Save and display the figure
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'subset_influence.pdf')
    plt.savefig(output_path)

if __name__ == "__main__":
    from user_grouping_evaluation import run_movie

    # Run the simulation to generate data
    df_stats, df_model, model_point_histories, expert_point, user_points, chosen_subsets = run_movie(
        num_users=10, num_experts=1, num_runs=1, num_groups=3, num_of_rounds=20
    )

    # Generate and display the subset influence plot using the simulation data
    plot_subset_influence_static(
        user_points,
        expert_point,
        model_point_histories,
        chosen_subsets,
        round_indices=[0, 6, 12, 19],
        title="How a User Subset Pulls the Model Toward Its Centroid"
    )
