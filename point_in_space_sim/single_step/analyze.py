import pathlib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import dotenv
import sys

dotenv.load_dotenv()

thisdir = pathlib.Path(__file__).parent.absolute()

def main(emotion_target):
    emotiondir = thisdir.joinpath(f"output/{emotion_target}")
    neighborhood = json.loads(emotiondir.joinpath(
        "neighborhood.json").read_text())
    output_sentences = json.loads(
        emotiondir.joinpath("all_sentences.json").read_text())

    # Compute the center of the neighborhood
    center = np.mean([sentence["emotion"]
                     for sentence in neighborhood[1:]], axis=0)

    rows = []
    neighborhood_distances = []
    for i, sentence in enumerate(neighborhood[1:], start=1):
        distance = np.linalg.norm(center - np.array(sentence["emotion"]))
        neighborhood_distances.append(distance)
        # We will not add neighborhood data to the DataFrame this time

    for i, sentence in enumerate(output_sentences, start=1):
        distance = np.linalg.norm(center - np.array(sentence["emotion"]))
        # Check the k_fraction and assign model names accordingly
        k_fraction = sentence.get('k_fraction', None)
        if k_fraction == 33:
            model_label = "Model Gen 1"
        elif k_fraction == 66:
            model_label = "Model Gen 2"
        elif k_fraction == 100:
            model_label = "Model Gen 3"

        rows.append([i, sentence["type"], model_label, distance])

    df = pd.DataFrame(rows, columns=["num", "type", "samples", "distance"])

    sns.set(style="whitegrid")  # Set the seaborn style

    plt.figure(figsize=(12, 6))
    palette = sns.color_palette("husl", 1)  # Color palette

    sns.boxplot(x="samples", y="distance", hue="type",
                data=df, palette=palette, width=0.25)

    # Calculate the 25th and 75th percentiles for the neighborhood distances
    lower_quartile = np.percentile(neighborhood_distances, 25)
    upper_quartile = np.percentile(neighborhood_distances, 75)
    plt.axhspan(lower_quartile, upper_quartile, color='lightgreen',
                alpha=0.3, label='Target Neighborhood')

    plt.title("Distance from Target Sentence with Neighborhood Percentile Band")
    plt.xlabel("Model Progression")
    plt.ylabel("Distance")
    plt.legend(title="Data Description", loc="upper right")

    # Save the plot
    savepath = emotiondir.joinpath("distance.png")
    savepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(savepath)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        emotion_target = sys.argv[1]
        
    main(emotion_target)
