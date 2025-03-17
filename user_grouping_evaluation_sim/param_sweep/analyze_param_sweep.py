import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
df = pd.read_csv('./param_sweep_stats/aggregated_stats.csv')

# Function to find best combinations
def find_best_combinations(df):
    # For Pearson Correlation, higher is better
    best_correlation = df.groupby(['num_users', 'num_groups']).apply(
        lambda x: x.loc[x['Pearson Correlation'].idxmax()]
    )[['num_users', 'num_groups', 'Grouping Function', 'Evaluation Function', 'Pearson Correlation']]
    
    # For Convergence Accuracy, lower is better (it's a distance metric)
    best_convergence = df.groupby(['num_users', 'num_groups']).apply(
        lambda x: x.loc[x['Convergence Accuracy (final distance)'].idxmin()]
    )[['Grouping Function', 'Evaluation Function', 'Convergence Accuracy (final distance)']]
    
    return best_correlation, best_convergence

# Updated function to count how many times each combination is the best,
# including the percentage based on total parameter combinations.
def count_best_occurrences(best_data, metric_name, total_param_combos):
    counts = best_data[['Grouping Function', 'Evaluation Function']].value_counts().reset_index()
    counts.columns = ['Grouping Function', 'Evaluation Function', f'Count (Best {metric_name})']
    counts[f'Percentage (Best {metric_name})'] = 100.0 * counts[f'Count (Best {metric_name})'] / total_param_combos
    counts = counts.sort_values(f'Count (Best {metric_name})', ascending=False)
    return counts

# Function to analyze how performance changes with user and group counts
def analyze_trends(df):
    # Create grouped data for analysis
    user_trends = df.groupby(['num_users', 'Grouping Function', 'Evaluation Function']).agg({
        'Pearson Correlation': 'mean',
        'Convergence Accuracy (final distance)': 'mean'
    }).reset_index()
    
    group_trends = df.groupby(['num_groups', 'Grouping Function', 'Evaluation Function']).agg({
        'Pearson Correlation': 'mean',
        'Convergence Accuracy (final distance)': 'mean'
    }).reset_index()
    
    return user_trends, group_trends

# Function to create a facet grid line plot for Pearson Correlation vs. number of users
def create_facetgrid_lineplot():
    # Load the aggregated CSV file
    csv_path = "./param_sweep_stats/aggregated_stats.csv"
    df = pd.read_csv(csv_path)
    
    # Rename columns for easier reference
    df = df.rename(columns={
        "Convergence Accuracy (final distance)": "final_distance",
        "Pearson Correlation": "pearson_corr"
    })
    df = df.dropna(subset=["final_distance", "pearson_corr"], how="any")
    df["Evaluation Function"] = df["Evaluation Function"].str.replace("_", " ").str.title()
    df["Grouping Function"] = df["Grouping Function"].str.replace("_", " ").str.title()

    # Create a FacetGrid:
    # - Columns: Grouping Function
    # - Rows: Evaluation Function
    # Within each facet, plot a line graph:
    #   x-axis: Number of Users (sorted)
    #   y-axis: Pearson Correlation
    #   Hue: Number of Groups (each line represents a different group count)
    g = sns.FacetGrid(
        df, 
        col="Grouping Function", 
        row="Evaluation Function", 
        margin_titles=True, 
        height=5,
        aspect=1.5
    )
    
    # Use lineplot instead of scatterplot:
    g.map_dataframe(
        sns.lineplot,
        x="num_users", 
        y="pearson_corr", 
        hue="num_groups", 
        marker="o",
        palette="Set1",
        linewidth=1.5,
        markersize=12
    )

    g.set_titles(col_template="{col_name}", row_template="{row_name}", size=28)

    g.add_legend(title="# of Groups", fontsize=24, loc='upper right', title_fontsize='24')
    if g._legend is not None:
        g._legend.get_title().set_fontsize(24)

    g.set_axis_labels("Number of Users", "Pearson Correlation", fontsize=28)
    g.figure.subplots_adjust(top=0.85, left=0.07, bottom=0.08, right=0.93)
    
    g.figure.suptitle("Pearson Correlation vs. Number of Users\nfor Different Group Sizes", fontsize=30, fontweight='bold')
    
    for ax in g.axes.flat:
        ax.set_xlabel(ax.get_xlabel(), fontsize=26)
        ax.set_ylabel(ax.get_ylabel(), fontsize=26)
        ax.tick_params(axis='both', labelsize=26)
        
    fig = g.figure
    col_pos = (g.axes[0, 0].get_position().x0 + g.axes[0, -1].get_position().x1) / 2
    fig.text(col_pos, 0.90, "Grouping Methods", ha='center', va='center', fontsize=28, fontstyle='italic')
    
    # Add "Evaluation Methods" title to the right of rows
    # Calculate the middle position of all rows
    row_pos = (g.axes[0, 0].get_position().y0 + g.axes[-1, 0].get_position().y1) / 2
    fig.text(0.98, row_pos, "Evaluation Methods", ha='center', va='center', fontsize=28, fontstyle='italic', rotation=270)

    plt.savefig("lineplot_users_by_groups.pdf", dpi=300)
    plt.close()

# Function to create a facet grid line plot for Final Distance vs. number of users
def create_facetgrid_lineplot_final_distance():
    # Load the aggregated CSV file
    csv_path = "./param_sweep_stats/aggregated_stats.csv"
    df = pd.read_csv(csv_path)
    
    # Rename columns for easier reference
    df = df.rename(columns={
        "Convergence Accuracy (final distance)": "final_distance",
        "Pearson Correlation": "pearson_corr"
    })
    df = df.dropna(subset=["final_distance", "pearson_corr"], how="any")
    
    df['Evaluation Function'] = df['Evaluation Function'].str.replace('_', ' ').str.title()
    df['Grouping Function'] = df['Grouping Function'].str.replace('_', ' ').str.title()

    # Create a FacetGrid:
    # - Columns: Grouping Function
    # - Rows: Evaluation Function
    # Within each facet, plot a line graph:
    #   x-axis: Number of Users (sorted)
    #   y-axis: Final Distance
    #   Hue: Number of Groups (each line represents a different group count)
    g = sns.FacetGrid(
        df, 
        col="Grouping Function", 
        row="Evaluation Function", 
        margin_titles=True, 
        height=5,
        aspect=1.5
    )
    
    # Use lineplot: final_distance on y-axis
    g.map_dataframe(
        sns.lineplot,
        x="num_users", 
        y="final_distance", 
        hue="num_groups", 
        marker="o",
        palette="Set1",
        linewidth=1.5,
        markersize=12,
    )

    g.set_titles(col_template="{col_name}", row_template="{row_name}", size=28)

    g.add_legend(title="# of Groups", fontsize=24, loc='upper right', title_fontsize='24')
    if g._legend is not None:
        g._legend.get_title().set_fontsize(24)

    g.set_axis_labels("Number of Users", "Final Distance", fontsize=28)
    g.figure.subplots_adjust(top=0.85, left=0.07, bottom=0.08, right=0.93)
    g.figure.suptitle("Final Distance vs. Number of Users\nfor Different Group Sizes", fontsize=30, fontweight='bold')

    for ax in g.axes.flat:
        ax.set_xlabel(ax.get_xlabel(), fontsize=26)
        ax.set_ylabel(ax.get_ylabel(), fontsize=26)
        ax.tick_params(axis='both', labelsize=26)
        
    fig = g.figure
    col_pos = (g.axes[0, 0].get_position().x0 + g.axes[0, -1].get_position().x1) / 2
    fig.text(col_pos, 0.90, "Grouping Methods", ha='center', va='center', fontsize=28, fontstyle='italic')
    
    # Add "Evaluation Methods" title to the right of rows
    # Calculate the middle position of all rows
    row_pos = (g.axes[0, 0].get_position().y0 + g.axes[-1, 0].get_position().y1) / 2
    fig.text(0.98, row_pos, "Evaluation Methods", ha='center', va='center', fontsize=28, fontstyle='italic', rotation=270)
    
    plt.savefig("lineplot_final_distance_users_by_groups.pdf", dpi=300)
    plt.close()

def plot_trend_for_method_combination(grouping_method, evaluation_method, metric, output_filename):
    """
    Plot the trend of a performance metric for a specified method combination
    across different numbers of users, with separate lines for different group counts.
    
    Parameters:
      grouping_method (str): Name of the Grouping Function (e.g., "random").
      evaluation_method (str): Name of the Evaluation Function (e.g., "l2_norm").
      metric (str): Performance metric to plot, either "pearson_corr" or "final_distance".
      output_filename (str): Filename for saving the plot (e.g., "trend_random_l2_pearson.pdf").
    """
    # Load the aggregated CSV file
    df = pd.read_csv('./param_sweep_stats/aggregated_stats.csv')
    
    # Rename columns for consistency
    df = df.rename(columns={
        "Convergence Accuracy (final distance)": "final_distance",
        "Pearson Correlation": "pearson_corr"
    })
    
    # Filter the data for the specified method combination (case-insensitive)
    mask = (
        df["Grouping Function"].str.lower() == grouping_method.lower()
    ) & (
        df["Evaluation Function"].str.lower() == evaluation_method.lower()
    )
    df_method = df[mask]
    
    if df_method.empty:
        print("No data found for the specified method combination.")
        return
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_method, x="num_users", y=metric, hue="num_groups",
                 marker="o", linewidth=2, palette="Set1")
    
    plt.xlabel("Number of Users", fontsize=14)
    if metric == "pearson_corr":
        plt.ylabel("Average Pearson Correlation", fontsize=14)
        plt.title(f"Trend in Pearson Correlation\nfor ({grouping_method}, {evaluation_method})", fontsize=16)
    elif metric == "final_distance":
        plt.ylabel("Average Final Distance", fontsize=14)
        plt.title(f"Trend in Final Distance\nfor ({grouping_method}, {evaluation_method})", fontsize=16)
    else:
        plt.ylabel(metric, fontsize=14)
        plt.title(f"Trend in {metric}\nfor ({grouping_method}, {evaluation_method})", fontsize=16)
    
    plt.legend(title="Number of Groups", fontsize=12, title_fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()


# Run the analysis
if __name__ == "__main__":    
    # Find best combinations for each metric
    best_correlation, best_convergence = find_best_combinations(df)
    
    # Determine total number of (num_users, num_groups) combinations
    total_param_combos = df["num_users"].nunique() * df["num_groups"].nunique()
    
    # Count occurrences of best combinations, now with percentages
    corr_counts = count_best_occurrences(best_correlation, 'Correlation', total_param_combos)
    conv_counts = count_best_occurrences(best_convergence, 'Convergence', total_param_combos)

    
    # Analyze trends
    user_trends, group_trends = analyze_trends(df)
    
    # Save results to CSV files
    best_correlation.to_csv('best_correlation_by_config.csv', index=False)
    best_convergence.to_csv('best_convergence_by_config.csv', index=False)
    corr_counts.to_csv('best_correlation_counts.csv', index=False)
    conv_counts.to_csv('best_convergence_counts.csv', index=False)
    
    # Create facet grid line plots
    create_facetgrid_lineplot()
    create_facetgrid_lineplot_final_distance()
