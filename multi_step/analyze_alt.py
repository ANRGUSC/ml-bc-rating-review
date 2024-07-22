import pathlib
import json
import numpy as np
import pandas as pd
import dotenv
import matplotlib.pyplot as plt
import os

# Load environment variables
dotenv.load_dotenv()

# Get the directory of the script being run
thisdir = pathlib.Path(__file__).parent.absolute()

emotion_target = "love"


def main():
    output_dir = thisdir.joinpath('output')
    emotions = [folder for folder in os.listdir(
        output_dir) if os.path.isdir(output_dir.joinpath(folder))]

    for emotion_target in emotions:
        emotion_dir = output_dir.joinpath(f"{emotion_target}")
        # Load data from the neighborhood JSON file
        with open(emotion_dir.joinpath('neighborhood.json'), 'r') as file:
            neighborhood = json.load(file)

        # Calculate the center of the neighborhood based on emotion vectors
        center = np.mean([sentence["emotion"]
                          for sentence in neighborhood[1:]], axis=0)

        # Initialize list to store data
        rows = []

        # Compute distances for neighborhood sentences
        for i, sentence in enumerate(neighborhood[1:], start=1):
            distance = np.linalg.norm(center - np.array(sentence["emotion"]))
            rows.append([i, "neighborhood", "N/A", distance])

        # Load and compute distances for multiple output sentences across generations
        for gen in range(1, 4):  # Assuming three generations for now
            output_sentences = json.loads(
                (emotion_dir.joinpath(f"all_sentences_{gen}.json")).read_text())

            for i, sentence in enumerate(output_sentences, start=1):
                distance = np.linalg.norm(
                    center - np.array(sentence["emotion"]))
                label = f"k={sentence['k_fraction']}" if sentence["type"] == "output-ft" else "N/A"
                rows.append(
                    [i, sentence["type"], f"{label} Gen {gen}", distance])

        # Create a DataFrame from the rows
        df = pd.DataFrame(rows, columns=["num", "type", "samples", "distance"])

        plt.figure(figsize=(12, 6))

        generations = [1, 2, 3]
        colors = {'mean': 'blue', '1st quartile': 'green',
                  '3rd quartile': 'red'}
        markers = {
            'mean': 'p',
            '1st quartile': 'h',
            '3rd quartile': 'D'
        }

        for gen in generations:
            gen_data = df[df['samples'].str.contains(f"Gen {gen}")]
            if not gen_data.empty:
                mean_distance = gen_data['distance'].mean()
                first_quartile = gen_data['distance'].quantile(0.25)
                third_quartile = gen_data['distance'].quantile(0.75)

                plt.scatter([gen], [mean_distance], color=colors['mean'],
                            marker=markers['mean'], s=100, label=f'Mean Gen {gen}' if gen == 1 else "")
                plt.scatter([gen], [first_quartile], color=colors['1st quartile'],
                            marker=markers['1st quartile'], s=100, label=f'1st Quartile Gen {gen}' if gen == 1 else "")
                plt.scatter([gen], [third_quartile], color=colors['3rd quartile'],
                            marker=markers['3rd quartile'], s=100, label=f'3rd Quartile Gen {gen}' if gen == 1 else "")

        # Filter out the neighborhood data for the plot
        neighborhood_data = df[df['type'] == 'neighborhood']['distance']

        # Calculate the range for the horizontal band
        lower_bound = neighborhood_data.quantile(0.25)
        upper_bound = neighborhood_data.quantile(0.75)

        # Add a horizontal band across the plot
        plt.axhspan(lower_bound, upper_bound, color='lightgreen', alpha=0.3)

        # Adjusting the aesthetics of the plot
        plt.title(f'Distance Distribution for {emotion_target}')
        plt.ylabel('Distance from Center')
        plt.xticks(generations, [f'Model Gen {gen}' for gen in generations])
        plt.grid(True)

        # Show the plot
        plt.legend(title='Type')
        plt.savefig(emotion_dir.joinpath(f'dist_alt.png'))

        plt.clf()
        plt.close()


if __name__ == "__main__":
    main()
