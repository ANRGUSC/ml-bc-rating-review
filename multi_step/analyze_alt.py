import pathlib
import json
import numpy as np
import os
import dotenv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from fine_tune import prompt, sentence_arrays, label_order
# from prepare import classifier

dotenv.load_dotenv()

thisdir = pathlib.Path(__file__).parent.absolute()


def main():
    neighborhood = json.loads(thisdir.joinpath(
        "neighborhood.json").read_text())
    output_sentences = json.loads(
        thisdir.joinpath("all_sentences.json").read_text())

    # Compute the center of the neighborhood
    center = np.mean([sentence["emotion"]
                     for sentence in neighborhood[1:]], axis=0)

    rows = []
    for i, sentence in enumerate(neighborhood[1:], start=1):
        distance = np.linalg.norm(center - np.array(sentence["emotion"]))
        rows.append([i, "neighborhood", "N/A", distance])

    for i, sentence in enumerate(output_sentences, start=1):
        distance = np.linalg.norm(center - np.array(sentence["emotion"]))
        label = f"k={sentence['k_fraction']}"
        rows.append([i, sentence["type"], label, distance])

    df = pd.DataFrame(rows, columns=["num", "type", "samples", "distance"])

    # Calculating the percentiles for the neighborhood distances
    neighborhood_distances = df[df['type'] == 'neighborhood']['distance']
    lower_bound = np.percentile(neighborhood_distances, 25)
    upper_bound = np.percentile(neighborhood_distances, 75)

    # Plotting using matplotlib
    plt.figure(figsize=(12, 6))
    sns.set(style="whitegrid")  # Setting the seaborn style

    # Prepare data for box plots
    unique_samples = df[df['type'] != 'neighborhood']['samples'].unique()
    data_to_plot = [df[(df['type'] != 'neighborhood') & (
        df['samples'] == sample)]['distance'] for sample in unique_samples]

    # Color palette
    palette = sns.color_palette("husl", len(unique_samples))
    sns.boxplot(data=data_to_plot, palette=palette)

    # Plot neighborhood band
    plt.axhspan(lower_bound, upper_bound, color='lightgreen',
                alpha=0.3, label='Neighborhood (25th-75th Percentile)')

    plt.title("Distance from Target Sentence with Neighborhood Percentile Band")
    plt.xlabel("Sample Group")
    plt.ylabel("Distance")
    plt.legend(title="Data Description")

    # Save the plot
    savepath = thisdir.joinpath("output/distance_styled.png")
    savepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(savepath)
    plt.show()


if __name__ == "__main__":
    main()
