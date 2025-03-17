import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import dotenv
import pathlib

dotenv.load_dotenv()

thisdir = pathlib.Path(__file__).parent.absolute()

def generate_emotion_improvement_stats(all_data, output_dir):
    """
    Calculates improvement statistics for the three specific emotions (disappointment, nervousness, neutral)
    between different generations and saves the results to a CSV file.
    
    Parameters:
    -----------
    all_data : list of pd.DataFrame
        List of DataFrames for each emotion.
    output_dir : pathlib.Path
        The output directory where the CSV will be saved.
    """
    # Concatenate all emotion dataframes
    all_df = pd.concat(all_data)
    
    # Define the specific emotions of interest
    selected_emotions = ["disappointment", "nervousness", "neutral"]
    
    # Filter the dataframe for the selected emotions
    stats_data = all_df[all_df['emotion'].isin(selected_emotions)]
    
    # Create a dataframe to store the statistics
    stats_rows = []
    
    for emotion in selected_emotions:
        emotion_data = stats_data[stats_data['emotion'] == emotion]
        
        # Calculate mean distances for each generation
        gen1_mean = emotion_data[emotion_data['generation'] == 1]['distance'].mean()
        gen2_mean = emotion_data[emotion_data['generation'] == 2]['distance'].mean()
        gen3_mean = emotion_data[emotion_data['generation'] == 3]['distance'].mean()
        
        # Calculate improvements as percentages
        gen1_to_gen2_improvement = ((gen1_mean - gen2_mean) / gen1_mean) * 100 if gen1_mean else 0
        gen2_to_gen3_improvement = ((gen2_mean - gen3_mean) / gen2_mean) * 100 if gen2_mean else 0
        gen1_to_gen3_improvement = ((gen1_mean - gen3_mean) / gen1_mean) * 100 if gen1_mean else 0
        
        # Add row to the stats dataframe
        stats_rows.append({
            'emotion': emotion,
            'gen1_mean_distance': gen1_mean,
            'gen2_mean_distance': gen2_mean,
            'gen3_mean_distance': gen3_mean,
            'gen1_to_gen2_improvement_pct': gen1_to_gen2_improvement,
            'gen2_to_gen3_improvement_pct': gen2_to_gen3_improvement,
            'gen1_to_gen3_improvement_pct': gen1_to_gen3_improvement
        })
    
    # Create and save the stats dataframe
    stats_df = pd.DataFrame(stats_rows)
    
    # Round numerical values for better readability
    numeric_cols = stats_df.select_dtypes(include=['float64']).columns
    stats_df[numeric_cols] = stats_df[numeric_cols].round(2)
    
    # Save to CSV
    csv_path = output_dir.joinpath('emotion_improvement_stats.csv')
    stats_df.to_csv(csv_path, index=False)
    
    print(f"Emotion improvement statistics saved to {csv_path}")
    
    return stats_df

def generate_specific_emotions_plot(all_data, emotion_percentiles, output_dir):
    """
    Generates a boxplot comparing the distance distributions for the emotions:
    'disappointment', 'nervousness', and 'neutral', and saves the figure as a PDF.
    
    Parameters:
    -----------
    all_data : list of pd.DataFrame
        List of DataFrames for each emotion.
    emotion_percentiles : dict
        Dictionary mapping each emotion to its (lower_bound, upper_bound) percentile.
    output_dir : pathlib.Path
        The output directory where the figure will be saved.
    """
    # Concatenate all emotion dataframes
    all_df = pd.concat(all_data)
    
    # Define the specific emotions of interest
    selected_emotions = ["disappointment", "nervousness", "neutral"]
    
    # Filter the dataframe for the selected emotions
    plot_data = all_df[all_df['emotion'].isin(selected_emotions)]
    
    sns.set_context("paper", font_scale=2)
    plt.rcParams.update({'font.size': 20})
    
    plt.figure(figsize=(12, 8))
    # Use a palette with exactly three colors (one per selected emotion)
    sns_palette = sns.color_palette("Set3", n_colors=len(selected_emotions))
    
    # Fix the order to match our selected_emotions list
    ax = sns.boxplot(x='generation', y='distance', hue='emotion',
                     data=plot_data, palette=sns_palette,
                     hue_order=selected_emotions, width=0.4, showfliers=False)
    
    # For each selected emotion, overlay the percentile band
    for emotion, color in zip(selected_emotions, sns_palette):
        if emotion in emotion_percentiles:
            lower_bound, upper_bound = emotion_percentiles[emotion]
            plt.axhspan(lower_bound, upper_bound, facecolor=color, alpha=0.75)
    
    plt.title("Multi-Model Performance for Best, Median, and\n Worst Single-Model Emotions", fontsize=20)
    plt.ylabel("Distance from Center", fontsize=20)
    plt.xlabel("Generation", fontsize=20)
    plt.grid(True)
    
    # Customize the legend with capitalized emotion names
    handles, _ = ax.get_legend_handles_labels()
    custom_labels = [emotion.capitalize() for emotion in selected_emotions]
    plt.legend(handles, custom_labels, title="Emotion", loc="upper right",
               bbox_to_anchor=(0.73, -0.16), ncol=2, borderaxespad=0.0,
               fontsize=16, title_fontsize=16)
    
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(output_dir.joinpath("specific_emotions_distance.pdf"))
    plt.clf()
    plt.close()

def main():
    output_dir = thisdir.joinpath('output')
    emotions = [folder for folder in os.listdir(
        output_dir) if os.path.isdir(output_dir.joinpath(folder))]

    gen3_distances = []
    all_data = []
    emotion_percentiles = {}

    for emotion_target in emotions:
        emotion_dir = output_dir.joinpath(emotion_target)

        with open(emotion_dir.joinpath('neighborhood.json'), 'r') as file:
            neighborhood = json.load(file)

        center = np.mean([np.array(sentence["emotion"])
                         for sentence in neighborhood], axis=0)

        rows = []

        neighborhood_distances = [np.linalg.norm(
            center - np.array(sentence["emotion"])) for sentence in neighborhood[1:]]

        for gen in range(1, 4):
            output_sentences = json.loads(
                (emotion_dir.joinpath(f"all_sentences_{gen}.json")).read_text())
            output_distances = [np.linalg.norm(
                center - np.array(sentence["emotion"])) for sentence in output_sentences]
            rows.extend([(idx, gen, dist, emotion_target) for idx, dist in zip(
                range(1, len(output_distances) + 1), output_distances)])

        df = pd.DataFrame(
            rows, columns=["num", "generation", "distance", "emotion"])

        gen3_mean_distance = df[df['generation'] == 3]['distance'].mean()
        gen3_distances.append(gen3_mean_distance)

        all_data.append(df)

        lower_bound = np.percentile(neighborhood_distances, 25)
        upper_bound = np.percentile(neighborhood_distances, 75)
        emotion_percentiles[emotion_target] = (lower_bound, upper_bound)


    all_df = pd.concat(all_data)

    best_emotion = emotions[np.argmin(gen3_distances)]
    median_emotion = emotions[np.argsort(
        gen3_distances)[len(gen3_distances)//2]]
    worst_emotion = emotions[np.argmax(gen3_distances)]

    plot_data = all_df[all_df['emotion'].isin(
        [best_emotion, median_emotion, worst_emotion])]

    sns.set_context("paper", font_scale=2)
    plt.rcParams.update({'font.size': 20})
    
    plt.figure(figsize=(12, 8))
    sns_palette = sns.color_palette(
        "Set3", n_colors=len(plot_data['emotion'].unique()))
    ax = sns.boxplot(x='generation', y='distance', hue='emotion',
                     data=plot_data, palette=sns_palette, hue_order=[worst_emotion, median_emotion, best_emotion], width=0.4, showfliers=False)

    emotion_color_mapping = {emotion: color for emotion, color in zip(
        [worst_emotion, median_emotion, best_emotion], sns_palette)}

    for emotion, color in emotion_color_mapping.items():
        lower_bound, upper_bound = emotion_percentiles[emotion]
        plt.axhspan(lower_bound, upper_bound, facecolor=color,
                    alpha=0.75)

    plt.title(f"Multi-Model Distance Distribution Comparison", fontsize=20)
    plt.ylabel('Distance from Center', fontsize=20)
    plt.xlabel('Generation', fontsize=20)
    plt.grid(True)
    handles, _ = ax.get_legend_handles_labels()
    custom_labels = [f'worst emotion ({worst_emotion})',
                     f'median emotion ({median_emotion})', f'best emotion ({best_emotion})']
    plt.legend(
        handles, 
        custom_labels,
        title='Emotion Performance', 
        loc='upper right', 
        bbox_to_anchor=(0.95, -0.16),
        ncol=2,
        borderaxespad=0.0,
        fontsize=16, 
        title_fontsize=16)

    plt.subplots_adjust(bottom=0.25)
    plt.savefig(output_dir.joinpath('combined_dist.pdf'))
    plt.clf()
    plt.close()
    
    generate_specific_emotions_plot(all_data, emotion_percentiles, output_dir)
    generate_emotion_improvement_stats(all_data, output_dir)

if __name__ == "__main__":
    main()