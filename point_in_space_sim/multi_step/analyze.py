import os
import sys
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import dotenv
import pathlib

dotenv.load_dotenv()
thisdir = pathlib.Path(__file__).parent.absolute()

def main(emotion_target, output_name):
    output_dir = thisdir.joinpath(output_name)
    emotion_dir = output_dir.joinpath(emotion_target)

    with open(emotion_dir.joinpath('neighborhood.json'), 'r') as file:
        neighborhood = json.load(file)

    center = np.mean([np.array(sentence["emotion"]) for sentence in neighborhood], axis=0)

    rows = []

    neighborhood_distances = [np.linalg.norm(
        center - np.array(sentence["emotion"])) for sentence in neighborhood[1:]]

    for gen in range(1, 4):
        json_path = emotion_dir.joinpath(f"all_sentences_{gen}.json")
        
        output_sentences = json.loads(json_path.read_text())
        output_distances = [np.linalg.norm(
            center - np.array(sentence["emotion"])) for sentence in output_sentences]
        
        rows.extend([(idx, f"Gen {gen}", dist, emotion_target) 
                     for idx, dist in zip(range(1, len(output_distances) + 1), output_distances)])

    df = pd.DataFrame(rows, columns=["num", "generation", "distance", "emotion"])

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='generation', y='distance', data=df, hue="generation", width=0.3, showfliers=False)

    lower_bound = np.percentile(neighborhood_distances, 25)
    upper_bound = np.percentile(neighborhood_distances, 75)

    plt.axhspan(lower_bound, upper_bound, color='lightgreen', alpha=0.3)

    plt.title(f'Distance Distribution for {emotion_target.capitalize()}')
    plt.ylabel('Distance from Center')
    plt.xlabel('Generation')
    plt.grid(True)
    plt.savefig(emotion_dir.joinpath('dist.png'))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        emotion_target = sys.argv[1]
        output_name = sys.argv[2]
    main(emotion_target, output_name)
