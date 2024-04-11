import pathlib
import json
from typing import List
import numpy as np
from openai import OpenAI
import os
import dotenv
import pandas as pd
import plotly.express as px

from collections import defaultdict
from fine_tune import generic_prompt, sentence_arrays, label_order
# from prepare import classifier

dotenv.load_dotenv()

thisdir = pathlib.Path(__file__).parent.absolute()

def categorize_love_scores(output_sentences):
    categories = [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
    counts = defaultdict(lambda: defaultdict(lambda: {f"{lower}-{upper}": 0 for lower, upper in categories}))
    
    for sentence in output_sentences:
        love_score = sentence["emotion"][0]
        sentence_type = sentence["type"]
        k_fraction = sentence.get("k_fraction", "output-pe")
        
        for lower, upper in categories:
            if lower <= love_score < upper:
                category = f"{lower}-{upper}"
                counts[sentence_type][k_fraction][category] += 1
                break

    return counts


def report_counts(counts):
    data = []
    for sentence_type, k_fractions in counts.items():
        for k_fraction, categories in k_fractions.items():
            for category, count in categories.items():
                label = k_fraction if sentence_type == "output-ft" else sentence_type
                data.append({"Type/K-Fraction": label, "Category": category, "Count": count})

    df_counts = pd.DataFrame(data)
    df_pivot = df_counts.pivot(index="Type/K-Fraction", columns="Category", values="Count").fillna(0)

    thisdir = pathlib.Path(__file__).parent.absolute()
    save_dir = thisdir.joinpath("emotion_trends")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    csv_file_path = save_dir.joinpath("emotion_trends.csv")
    df_pivot.to_csv(csv_file_path)

def save_average_love_scores_to_csv(output_sentences: list):
    df_sentences = pd.DataFrame(output_sentences)
    
    df_sentences["love_score"] = df_sentences["emotion"].apply(lambda x: x[0])
    
    average_output_ft = df_sentences[df_sentences['type'] == 'output-ft'].groupby('k_fraction')['love_score'].mean().reset_index()
    
    average_output_pe = df_sentences[df_sentences['type'] == 'output-pe']["love_score"].mean()
    
    average_output_ft['k_fraction'] = average_output_ft['k_fraction'].astype(int)
    pe_row = pd.DataFrame([{'k_fraction': 'output-pe', 'love_score': average_output_pe}])
    
    average_combined = pd.concat([average_output_ft, pe_row], ignore_index=True)
    
    average_combined.columns = ['k_fraction', 'love_score_average']
    
    thisdir = pathlib.Path(__file__).parent.absolute()
    save_dir = thisdir.joinpath("emotion_trends")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    csv_file_path = save_dir.joinpath("averages.csv")
    average_combined.to_csv(csv_file_path, index=False)

def save_average_distances_to_csv(output_sentences: list, center: np.ndarray):
    df_sentences = pd.DataFrame(output_sentences)
    df_sentences["distance"] = df_sentences["emotion"].apply(lambda emotions: np.linalg.norm(center - np.array(emotions)))
    
    average_distances_ft = df_sentences[df_sentences['type'] == 'output-ft'].groupby('k_fraction')['distance'].mean().reset_index()
    
    average_distance_pe = df_sentences[df_sentences['type'] == 'output-pe']["distance"].mean()
    
    average_distances_ft['k_fraction'] = average_distances_ft['k_fraction'].astype(int)
    pe_row = pd.DataFrame([{'k_fraction': 'output-pe', 'distance': average_distance_pe}])
    
    average_distances_combined = pd.concat([average_distances_ft, pe_row], ignore_index=True)
    
    average_distances_combined.columns = ['k_fraction', 'average_distance']
    
    thisdir = pathlib.Path(__file__).parent.absolute()
    save_dir = thisdir.joinpath("emotion_trends")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    csv_file_path = save_dir.joinpath("average_distances.csv")
    average_distances_combined.to_csv(csv_file_path, index=False)
    
def main():
    neighborhood = json.loads(thisdir.joinpath("neighborhood.json").read_text())
    output_sentences = json.loads(thisdir.joinpath("all_sentences.json").read_text())

    center = np.mean([sentence["emotion"] for sentence in neighborhood[1:]], axis=0)

    rows = []
    for i, sentence in enumerate(neighborhood[1:], start=1):
        distance = np.linalg.norm(center - np.array(sentence["emotion"]))
        rows.append([i, "neighborhood", "N/A", distance])

    for i, sentence in enumerate(output_sentences, start=1):
        distance = np.linalg.norm(center - np.array(sentence["emotion"]))
        label = f"k={sentence['k_fraction']}" if sentence["type"] == "output-ft" else "N/A"
        rows.append([i, sentence["type"], label, distance])

    df = pd.DataFrame(rows, columns=["num", "type", "samples", "distance"])
    print(df)
    fig = px.box(
        df,
        x="samples",
        y="distance",
        color="type",
        title="Distance from target sentence",
        points="all",
        template="plotly_white",
    )
    savepath = thisdir.joinpath("output/distance_default_loving_ft_prompt.html")
    savepath.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(savepath)
    
    counts = categorize_love_scores(output_sentences)
    report_counts(counts)
    
    save_average_love_scores_to_csv(output_sentences)
    save_average_distances_to_csv(output_sentences, center)

if __name__ == "__main__":
    main()    
