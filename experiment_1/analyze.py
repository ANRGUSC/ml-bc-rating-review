import pathlib
import json
from typing import List
import numpy as np
from openai import OpenAI
import os
import dotenv
import pandas as pd
import plotly.express as px
from fine_tune import lr, emotion_target

from collections import defaultdict

dotenv.load_dotenv()

thisdir = pathlib.Path(__file__).parent.absolute()


def plot_metric_trends(emotion_target: str, metric_type: str):
    # Define the range of learning rates, incremented by 0.25 from 1.00 to 3.00
    learning_rates = [i / 4 for i in range(4, 13)]
    frames = []
    for lr in learning_rates:
        file_path = pathlib.Path(
            f"emotion_trends/{emotion_target}/{lr:.2f}lr/{metric_type}.csv")
        if file_path.exists():
            df = pd.read_csv(file_path)
            df['learning_rate'] = lr
            frames.append(df)

    if not frames:
        print("No data files found for the specified parameters.")
        return

    df_concat = pd.concat(frames)

    y_axis_label = "average_distance" if 'distances' in metric_type else "score_average"

    fig = px.line(
        df_concat,
        x="learning_rate",
        y=y_axis_label,
        color="k_fraction",
        labels={"k_fraction": "K Fraction", "learning_rate": "Learning Rate",
                y_axis_label: y_axis_label.capitalize().replace('_', ' ')},
        title=f"Learning Rate vs {y_axis_label.split('_')[1].capitalize()} for {emotion_target}"
    )

    # Save the figure to an HTML file
    save_path = pathlib.Path(
        f"emotion_trends/{emotion_target}/{metric_type}_lr_plot.html")
    # Ensure the directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(save_path))


def categorize_scores(output_sentences):
    categories = [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
    counts = defaultdict(lambda: defaultdict(
        lambda: {f"{lower}-{upper}": 0 for lower, upper in categories}))
    total_counts = defaultdict(lambda: defaultdict(
        lambda: {f"{lower}-{upper}": 0 for lower, upper in categories}))

    for sentence in output_sentences:
        score = sentence["emotion"][0]
        sentence_type = sentence["type"]
        k_fraction = sentence.get("k_fraction", "output-pe")

        for lower, upper in categories:
            if lower <= score < upper:
                category = f"{lower}-{upper}"
                counts[sentence_type][k_fraction][category] += 1
                total_counts[sentence_type][k_fraction][category] += 1
                break

    return counts


def report_counts(counts):
    data = []
    for sentence_type, k_fractions in counts.items():
        for k_fraction, categories in k_fractions.items():
            for category, count in categories.items():
                label = k_fraction if sentence_type == "output-ft" else sentence_type
                data.append({"Type/K-Fraction": label,
                            "Category": category, "Count": count})

    df_counts = pd.DataFrame(data)
    df_pivot = df_counts.pivot(
        index="Type/K-Fraction", columns="Category", values="Count").fillna(0)

    thisdir = pathlib.Path(__file__).parent.absolute()

    save_dir = thisdir.joinpath(f"emotion_trends/{emotion_target}/{lr:.2f}lr")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    csv_file_path = save_dir.joinpath("emotion_trends.csv")
    df_pivot.to_csv(csv_file_path)


def save_average_scores_to_csv(output_sentences: list):
    df_sentences = pd.DataFrame(output_sentences)

    df_sentences["score"] = df_sentences["emotion"].apply(lambda x: x[0])

    average_output_ft = df_sentences[df_sentences['type'] ==
                                     'output-ft'].groupby('k_fraction')['score'].mean().reset_index()

    average_output_pe = df_sentences[df_sentences['type']
                                     == 'output-pe']["score"].mean()

    average_output_ft['k_fraction'] = average_output_ft['k_fraction'].astype(
        int)
    pe_row = pd.DataFrame(
        [{'k_fraction': 'output-pe', 'score': average_output_pe}])

    average_combined = pd.concat(
        [average_output_ft, pe_row], ignore_index=True)

    average_combined.columns = ['k_fraction', 'score_average']

    thisdir = pathlib.Path(__file__).parent.absolute()
    save_dir = thisdir.joinpath(f"emotion_trends/{emotion_target}/{lr:.2f}lr")
    save_dir.mkdir(parents=True, exist_ok=True)

    csv_file_path = save_dir.joinpath("averages.csv")
    average_combined.to_csv(csv_file_path, index=False)


def save_average_distances_to_csv(output_sentences: list, center: np.ndarray):
    df_sentences = pd.DataFrame(output_sentences)
    df_sentences["distance"] = df_sentences["emotion"].apply(
        lambda emotions: np.linalg.norm(center - np.array(emotions)))

    average_distances_ft = df_sentences[df_sentences['type'] == 'output-ft'].groupby(
        'k_fraction')['distance'].mean().reset_index()

    average_distance_pe = df_sentences[df_sentences['type']
                                       == 'output-pe']["distance"].mean()

    average_distances_ft['k_fraction'] = average_distances_ft['k_fraction'].astype(
        int)
    pe_row = pd.DataFrame(
        [{'k_fraction': 'output-pe', 'distance': average_distance_pe}])

    average_distances_combined = pd.concat(
        [average_distances_ft, pe_row], ignore_index=True)

    average_distances_combined.columns = ['k_fraction', 'average_distance']

    thisdir = pathlib.Path(__file__).parent.absolute()
    save_dir = thisdir.joinpath(f"emotion_trends/{emotion_target}/{lr:.2f}lr")
    save_dir.mkdir(parents=True, exist_ok=True)

    csv_file_path = save_dir.joinpath("average_distances.csv")
    average_distances_combined.to_csv(csv_file_path, index=False)


def main():
    neighborhood = json.loads(thisdir.joinpath(
        "neighborhood.json").read_text())
    output_sentences = json.loads(
        thisdir.joinpath("all_sentences.json").read_text())

    center = np.mean([sentence["emotion"]
                     for sentence in neighborhood[1:]], axis=0)

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
    savepath = thisdir.joinpath(
        f"output/{emotion_target}/distance_{lr:.2f}lr_100k.html")
    savepath.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(savepath)

    counts = categorize_scores(output_sentences)
    report_counts(counts)

    save_average_scores_to_csv(output_sentences)
    save_average_distances_to_csv(output_sentences, center)


    # plot_metric_trends(emotion_target=emotion_target, metric_type="averages")
if __name__ == "__main__":
    main()
