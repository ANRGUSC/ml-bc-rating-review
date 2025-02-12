import pathlib
import json
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

    plt.title('Distance Distribution Comparison')
    plt.ylabel('Distance from Center')
    plt.xlabel('Number of Fine-Tuning Samples')
    plt.grid(True)
    plt.legend(title='Emotion Performance', loc='upper right')

    plt.savefig(output_dir.joinpath('combined_distance.png'))
    plt.clf()
    plt.close()

def generate_disappointment_optimism_gratitude_plot(all_data, emotion_percentiles, output_dir):
    all_df = pd.concat(all_data)

    # Select specific emotions
    selected_emotions = ["disappointment", "optimism", "gratitude"]
    plot_data = all_df[all_df['emotion'].isin(selected_emotions)]

    plt.figure(figsize=(12, 8))
    sns_palette = sns.color_palette("Set2", n_colors=len(plot_data['emotion'].unique()))
    
    ax = sns.boxplot(x='samples', y='distance', hue='emotion',
                     data=plot_data, palette=sns_palette, 
                     hue_order=selected_emotions, 
                     width=0.4, showfliers=False)

    # Get colors assigned to each emotion in the boxplot
    emotion_color_mapping = {emotion: color for emotion, color in zip(selected_emotions, sns_palette)}

    # Plot percentile bands for each selected emotion with matching colors and no legend entry
    for emotion, color in emotion_color_mapping.items():
        if emotion in emotion_percentiles:
            lower_bound, upper_bound = emotion_percentiles[emotion]
            plt.axhspan(lower_bound, upper_bound, facecolor=color, alpha=0.3, label="_nolegend_")  # Prevents from appearing in legend

    plt.title('Distance Distribution: Disappointment, Optimism, and Gratitude')
    plt.ylabel('Distance from Center')
    plt.xlabel('Number of Fine-Tuning Samples')
    plt.grid(True)
    plt.legend(title='Emotions', loc='upper right')

    plt.savefig(output_dir.joinpath('combined_distance_disappointment_optimism_gratitude.png'))
    plt.clf()
    plt.close()

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
    generate_comparison_plot(all_data, gen3_distances, emotions, emotion_percentiles, output_dir)

    # Generate additional comparison plot for disappointment, disgust, gratitude
    generate_disappointment_optimism_gratitude_plot(all_data, emotion_percentiles, output_dir)

if __name__ == "__main__":
    main()
