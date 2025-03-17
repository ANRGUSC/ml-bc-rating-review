import pathlib
import json
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import dotenv

dotenv.load_dotenv()

thisdir = pathlib.Path(__file__).parent.absolute()

def process_emotion(emotion_target, all_data, gen3_distances, emotion_percentiles):
    emotiondir = thisdir.joinpath(f"output/{emotion_target}")

    neighborhood = json.loads(emotiondir.joinpath("neighborhood.json").read_text())
    output_sentences = json.loads(emotiondir.joinpath("all_sentences.json").read_text())

    # Compute the center of the neighborhood
    center = np.mean([sentence["emotion"] for sentence in neighborhood[1:]], axis=0)

    rows = []
    neighborhood_distances = []
    
    for i, sentence in enumerate(neighborhood[1:], start=1):
        distance = np.linalg.norm(center - np.array(sentence["emotion"]))
        neighborhood_distances.append(distance)

    for i, sentence in enumerate(output_sentences, start=1):
        distance = np.linalg.norm(center - np.array(sentence["emotion"]))

        # Assign model labels based on k_fraction
        k_fraction = sentence.get('k_fraction', None)
        if k_fraction == 33:
            model_label = "33 Samples"
        elif k_fraction == 66:
            model_label = "66 Samples"
        elif k_fraction == 100:
            model_label = "100 Samples"
        else:
            model_label = "Unknown"

        rows.append([i, sentence["type"], model_label, distance, emotion_target])

    df = pd.DataFrame(rows, columns=["num", "type", "samples", "distance", "emotion"])
    
    # Store Gen 3 mean distance
    gen3_mean_distance = df[df["samples"] == "100 Samples"]["distance"].mean()
    gen3_distances.append(gen3_mean_distance)
    all_data.append(df)

    # Calculate 25th and 75th percentiles for this emotion
    lower_quartile = np.percentile(neighborhood_distances, 25)
    upper_quartile = np.percentile(neighborhood_distances, 75)
    emotion_percentiles[emotion_target] = (lower_quartile, upper_quartile)

def generate_comparison_plot(all_data, gen3_distances, emotions, emotion_percentiles, output_dir):
    all_df = pd.concat(all_data)

    # Determine best, median, and worst emotions
    best_emotion = emotions[np.argmin(gen3_distances)]
    median_emotion = emotions[np.argsort(gen3_distances)[len(gen3_distances)//2]]
    worst_emotion = emotions[np.argmax(gen3_distances)]

    plot_data = all_df[all_df['emotion'].isin([best_emotion, median_emotion, worst_emotion])]

    sns.set_context("paper", font_scale=2)
    plt.rcParams.update({'font.size': 20})
    
    plt.figure(figsize=(12, 8))
    sns_palette = sns.color_palette("Set3", n_colors=len(plot_data['emotion'].unique()))
    
    ax = sns.boxplot(x='samples', y='distance', hue='emotion',
                     data=plot_data, palette=sns_palette, 
                     hue_order=[worst_emotion, median_emotion, best_emotion], 
                     width=0.4, showfliers=False)

    # Get colors assigned to each emotion in the boxplot
    emotion_color_mapping = {emotion: color for emotion, color in zip([worst_emotion, median_emotion, best_emotion], sns_palette)}

    # Plot percentile bands for each emotion with matching colors and no legend entry
    for emotion, color in emotion_color_mapping.items():
        lower_bound, upper_bound = emotion_percentiles[emotion]
        plt.axhspan(lower_bound, upper_bound, facecolor=color, alpha=0.3, label="_nolegend_")  # Prevents from appearing in legend

    plt.title('Single Model Distance Distribution Comparison', fontsize=20)
    plt.ylabel('Distance from Center', fontsize=20)
    plt.xlabel('Number of Fine-Tuning Samples', fontsize=20)
    plt.grid(True)
    handles, _ = ax.get_legend_handles_labels()
    custom_labels = [f'worst emotion ({worst_emotion})',
                     f'median emotion ({median_emotion})',
                     f'best emotion ({best_emotion})']
    
    plt.legend(
        handles, 
        custom_labels, 
        title='Emotion Performance', 
        bbox_to_anchor=(0.9, -0.16),
        ncol=2,
        borderaxespad=0.0,
        fontsize=16, 
        title_fontsize=16
    )
    
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(output_dir.joinpath('combined_distance.pdf'))
    plt.clf()
    plt.close()
    
    return best_emotion, median_emotion, worst_emotion

def generate_emotion_improvement_stats(all_data, best_emotion, median_emotion, worst_emotion, output_dir):
    """
    Calculates improvement statistics for the best, median, and worst emotions
    between different sample sizes and saves the results to a CSV file.
    
    Parameters:
    -----------
    all_data : list of pd.DataFrame
        List of DataFrames for each emotion.
    best_emotion : str
        The emotion with the best performance.
    median_emotion : str
        The emotion with median performance.
    worst_emotion : str
        The emotion with the worst performance.
    output_dir : pathlib.Path
        The output directory where the CSV will be saved.
    """
    # Concatenate all emotion dataframes
    all_df = pd.concat(all_data)
    
    # Filter the dataframe for the selected emotions
    selected_emotions = [best_emotion, median_emotion, worst_emotion]
    stats_data = all_df[all_df['emotion'].isin(selected_emotions)]
    
    # Create a dataframe to store the statistics
    stats_rows = []
    
    for emotion in selected_emotions:
        emotion_data = stats_data[stats_data['emotion'] == emotion]
        
        # Calculate mean distances for each sample size
        samples_33_mean = emotion_data[emotion_data['samples'] == "33 Samples"]['distance'].mean()
        samples_66_mean = emotion_data[emotion_data['samples'] == "66 Samples"]['distance'].mean()
        samples_100_mean = emotion_data[emotion_data['samples'] == "100 Samples"]['distance'].mean()
        
        # Calculate improvements as percentages
        samples_33_to_66_improvement = ((samples_33_mean - samples_66_mean) / samples_33_mean) * 100 if samples_33_mean else 0
        samples_66_to_100_improvement = ((samples_66_mean - samples_100_mean) / samples_66_mean) * 100 if samples_66_mean else 0
        samples_33_to_100_improvement = ((samples_33_mean - samples_100_mean) / samples_33_mean) * 100 if samples_33_mean else 0
        
        # Add row to the stats dataframe
        stats_rows.append({
            'emotion': emotion,
            'emotion_type': 'best' if emotion == best_emotion else ('median' if emotion == median_emotion else 'worst'),
            'samples_33_mean_distance': samples_33_mean,
            'samples_66_mean_distance': samples_66_mean,
            'samples_100_mean_distance': samples_100_mean,
            'samples_33_to_66_improvement_pct': samples_33_to_66_improvement,
            'samples_66_to_100_improvement_pct': samples_66_to_100_improvement,
            'samples_33_to_100_improvement_pct': samples_33_to_100_improvement
        })
    
    # Create and save the stats dataframe
    stats_df = pd.DataFrame(stats_rows)
    
    # Sort by emotion type for better readability
    stats_df['emotion_type_order'] = stats_df['emotion_type'].map({'best': 1, 'median': 2, 'worst': 3})
    stats_df = stats_df.sort_values('emotion_type_order').drop('emotion_type_order', axis=1)
    
    # Round numerical values for better readability
    numeric_cols = stats_df.select_dtypes(include=['float64']).columns
    stats_df[numeric_cols] = stats_df[numeric_cols].round(2)
    
    # Save to CSV
    csv_path = output_dir.joinpath('performance_improvement_stats.csv')
    stats_df.to_csv(csv_path, index=False)
    
    print(f"Performance improvement statistics saved to {csv_path}")
    
    return stats_df

def main():
    output_dir = thisdir.joinpath("output")

    # Find all emotion folders (directories)
    emotions = [folder for folder in os.listdir(output_dir) if os.path.isdir(output_dir.joinpath(folder))]

    all_data = []
    gen3_distances = []
    emotion_percentiles = {}  # Store percentile bands for each emotion

    for emotion in emotions:
        process_emotion(emotion, all_data, gen3_distances, emotion_percentiles)

    # Generate final comparison plot for best, median, worst emotions
    best_emotion, median_emotion, worst_emotion = generate_comparison_plot(all_data, gen3_distances, emotions, emotion_percentiles, output_dir)

    generate_emotion_improvement_stats(all_data, best_emotion, median_emotion, worst_emotion, output_dir)

if __name__ == "__main__":
    main()
