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


def main():
    title = "Adaptive Shrinking Segmentation"
    output_dir = thisdir.joinpath('output')
    emotions = [folder for folder in os.listdir(
        output_dir) if os.path.isdir(output_dir.joinpath(folder))]

    gen3_distances = []
    all_data = []
    emotion_percentiles = {}
    mean_distance_data = []

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
            rows.extend([(idx, f"Gen {gen}", dist, emotion_target) for idx, dist in zip(
                range(1, len(output_distances) + 1), output_distances)])
            
            mean_distance = np.mean(output_distances)
            mean_distance_data.append({
                "generation": f"Gen {gen}",
                "mean_distance": mean_distance,
                "emotion": emotion_target
            })

        df = pd.DataFrame(
            rows, columns=["num", "generation", "distance", "emotion"])

        gen3_mean_distance = df[df['generation'] == 'Gen 3']['distance'].mean()
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

    plt.title(f"Distance Distribution Comparison - {title}")
    plt.ylabel('Distance from Center')
    plt.xlabel('Generation')
    plt.grid(True)
    handles, _ = ax.get_legend_handles_labels()
    custom_labels = [f'worst emotion ({worst_emotion})',
                     f'median emotion ({median_emotion})', f'best emotion ({best_emotion})']
    plt.legend(handles, custom_labels,
               title='Emotion Performance', loc='upper right')
    plt.savefig(output_dir.joinpath('combined_dist.png'))
    plt.clf()
    plt.close()

    mean_distance_df = pd.DataFrame(mean_distance_data)
    pivot_df = mean_distance_df.pivot(index='emotion', columns='generation', values='mean_distance')
    csv_path = output_dir.joinpath('mean_distance.csv')

    with open(csv_path, 'w', encoding='utf-8') as file:
        file.write(f"# {title}\n")
        pivot_df.to_csv(file)


if __name__ == "__main__":
    main()