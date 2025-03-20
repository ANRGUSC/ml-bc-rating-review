import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'param_sweep_stats', 'aggregated_stats.csv')

line_plots_dir = os.path.join('user_grouping_evaluation_sim', 'param_sweep', 'line_plots')
stats_dir = os.path.join('user_grouping_evaluation_sim', 'param_sweep', 'best_counts_stats')

os.makedirs(line_plots_dir, exist_ok=True)
os.makedirs(stats_dir, exist_ok=True)

# Read the CSV file
df = pd.read_csv(csv_path)

# Function to find best combinations
def find_best_combinations(df):
    # For Pearson Correlation, higher is better
    best_correlation = df.groupby(['num_users', 'num_groups']).apply(
        lambda x: x.loc[x['Pearson Correlation'].idxmax()],
        include_groups=False
    ).reset_index()
    
    # For Convergence Accuracy, lower is better (it's a distance metric)
    best_convergence = df.groupby(['num_users', 'num_groups']).apply(
        lambda x: x.loc[x['Convergence Accuracy (final distance)'].idxmin()],
        include_groups=False
    ).reset_index()
    
    return best_correlation, best_convergence

# Function to count how many times each combination is the best, based on total parameter combinations.
def count_best_occurrences(best_data, metric_name, total_param_combos):
    counts = best_data[['Grouping Function', 'Evaluation Function']].value_counts().reset_index()
    counts.columns = ['Grouping Function', 'Evaluation Function', f'Count (Best {metric_name})']
    counts[f'Percentage (Best {metric_name})'] = 100.0 * counts[f'Count (Best {metric_name})'] / total_param_combos
    counts = counts.sort_values(f'Count (Best {metric_name})', ascending=False)
    return counts

# Function to analyze how performance changes with user and group counts
def analyze_trends(df):
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
    df = pd.read_csv(csv_path)
    
    df = df.rename(columns={
        "Convergence Accuracy (final distance)": "final_distance",
        "Pearson Correlation": "pearson_corr"
    })
    df = df.dropna(subset=["final_distance", "pearson_corr"], how="any")
    df["Evaluation Function"] = df["Evaluation Function"].str.replace("_", " ").str.title()
    df["Grouping Function"] = df["Grouping Function"].str.replace("_", " ").str.title()

    g = sns.FacetGrid(
        df, 
        col="Grouping Function", 
        row="Evaluation Function", 
        margin_titles=True, 
        height=5,
        aspect=1.5
    )
    
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
    
    row_pos = (g.axes[0, 0].get_position().y0 + g.axes[-1, 0].get_position().y1) / 2
    fig.text(0.98, row_pos, "Evaluation Methods", ha='center', va='center', fontsize=28, fontstyle='italic', rotation=270)

    output_path = os.path.join(line_plots_dir, "lineplot_pearson_users_by_groups.pdf")
    plt.savefig(output_path, dpi=300)
    plt.close()

# Function to create a facet grid line plot for Final Distance vs. number of users
def create_facetgrid_lineplot_final_distance():
    df = pd.read_csv(csv_path)
    
    df = df.rename(columns={
        "Convergence Accuracy (final distance)": "final_distance",
        "Pearson Correlation": "pearson_corr"
    })
    df = df.dropna(subset=["final_distance", "pearson_corr"], how="any")
    
    df['Evaluation Function'] = df['Evaluation Function'].str.replace('_', ' ').str.title()
    df['Grouping Function'] = df['Grouping Function'].str.replace('_', ' ').str.title()

    g = sns.FacetGrid(
        df, 
        col="Grouping Function", 
        row="Evaluation Function", 
        margin_titles=True, 
        height=5,
        aspect=1.5
    )
    
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
    
    row_pos = (g.axes[0, 0].get_position().y0 + g.axes[-1, 0].get_position().y1) / 2
    fig.text(0.98, row_pos, "Evaluation Methods", ha='center', va='center', fontsize=28, fontstyle='italic', rotation=270)
    
    output_path = os.path.join(line_plots_dir, "lineplot_final_distance_users_by_groups.pdf")
    plt.savefig(output_path, dpi=300)
    plt.close()

if __name__ == "__main__":    
    best_correlation, best_convergence = find_best_combinations(df)
    
    total_param_combos = df["num_users"].nunique() * df["num_groups"].nunique()
    
    corr_counts = count_best_occurrences(best_correlation, 'Correlation', total_param_combos)
    conv_counts = count_best_occurrences(best_convergence, 'Convergence', total_param_combos)

    user_trends, group_trends = analyze_trends(df)
    
    corr_counts_path = os.path.join(stats_dir, 'best_correlation_counts.csv')
    conv_counts_path = os.path.join(stats_dir, 'best_convergence_counts.csv')

    corr_counts.to_csv(corr_counts_path, index=False)
    conv_counts.to_csv(conv_counts_path, index=False)
    
    create_facetgrid_lineplot()
    create_facetgrid_lineplot_final_distance()
