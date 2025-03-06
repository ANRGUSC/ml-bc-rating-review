import os
import json
import numpy as np
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns

# Define the directories for single-step and multi-step
single_step_dir = pathlib.Path("single_step/output")
multi_step_dir = pathlib.Path("multi_step/output")

# Find all emotion folders (directories)
emotions = [folder for folder in os.listdir(single_step_dir) if os.path.isdir(single_step_dir.joinpath(folder))]

# Initialize an empty list for the table
data = []

# Process each emotion
for emotion in emotions:
    single_emotion_dir = single_step_dir.joinpath(emotion)
    multi_emotion_dir = multi_step_dir.joinpath(emotion)

    # Load neighborhood data (same for both)
    neighborhood = json.loads(single_emotion_dir.joinpath("neighborhood.json").read_text())

    # Compute the center of the neighborhood
    center = np.mean([sentence["emotion"] for sentence in neighborhood[1:]], axis=0)

    # Process Single-Step: Extract distance for k_fraction == 100
    single_sentences = json.loads(single_emotion_dir.joinpath("all_sentences.json").read_text())
    single_step_distances = [
        np.linalg.norm(center - np.array(sentence["emotion"]))
        for sentence in single_sentences if sentence.get("k_fraction") == 100
    ]
    single_step_distance = np.mean(single_step_distances) if single_step_distances else None

    # Process Multi-Step: Extract distance from all_sentences_3.json
    multi_sentences = json.loads(multi_emotion_dir.joinpath("all_sentences_3.json").read_text())
    multi_step_distances = [
        np.linalg.norm(center - np.array(sentence["emotion"]))
        for sentence in multi_sentences
    ]
    multi_step_distance = np.mean(multi_step_distances) if multi_step_distances else None

    # Append results to the table
    data.append([emotion, single_step_distance, multi_step_distance])

# Create a DataFrame
df_results = pd.DataFrame(data, columns=["Emotion", "Single-Step Distance", "Multi-Step Distance"])

df_results["Improvement (%)"] = (
    (df_results["Single-Step Distance"] - df_results["Multi-Step Distance"]) / df_results["Single-Step Distance"]
) * 100

# Handle cases where Single-Step Distance is zero to avoid division errors
df_results["Improvement (%)"] = df_results["Improvement (%)"].replace([np.inf, -np.inf], np.nan)

df_results.to_csv("comparison_results.csv", index=False)

# Define file paths for saving the plots
grouped_bar_chart_path = "single_vs_multi_distance.png"
improvement_bar_chart_path = "multi_step_improvement.png"

# Set up figure for grouped bar chart (Single-Step vs Multi-Step Distance)
plt.figure(figsize=(14, 6))
colors = sns.color_palette("Set1", 2)
sns.barplot(x="Emotion", y="Single-Step Distance", data=df_results, color=colors[0], label="Single-Step")
sns.barplot(x="Emotion", y="Multi-Step Distance", data=df_results, color=colors[1], label="Multi-Step")

# Formatting
plt.xticks(rotation=90)
plt.ylabel("Distance to Target Neighborhood")
plt.xlabel("Emotion")
plt.title("Single-Step vs Multi-Step Distance Comparison in the Final Iteration")
plt.legend()
plt.grid(axis="y")

# Save the plot
plt.savefig(grouped_bar_chart_path, bbox_inches="tight")
plt.clf()

# Set up figure for percentage improvement bar chart
plt.figure(figsize=(14, 6))
sns.barplot(x="Emotion", y="Improvement (%)", data=df_results)

# Formatting
plt.xticks(rotation=90)
plt.ylabel("Improvement (%)")
plt.xlabel("Emotion")
plt.title("Percentage Improvement of Multi-Step Over Single-Step")
plt.axhline(0, color="black", linewidth=1)  # Reference line at 0%
plt.grid(axis="y")

# Save the plot
plt.savefig(improvement_bar_chart_path, bbox_inches="tight")
plt.clf()

# Return file paths to the user
grouped_bar_chart_path, improvement_bar_chart_path
