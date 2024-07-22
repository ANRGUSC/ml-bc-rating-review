
import umap
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# CODE: UMAP first and then centroid compute
directory_path = 'output'
emotions = [d for d in os.listdir(directory_path) if os.path.isdir(
    os.path.join(directory_path, d))]

group_1 = emotions[:10]
group_2 = emotions[10:20]
group_3 = emotions[20:]


def plot_emotion_group_centroids(emotions, title):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1,
                        n_components=2, metric='euclidean')
    plt.figure(figsize=(12, 10))
    sns.set_theme(style="whitegrid")

    colors = sns.color_palette("hsv", len(emotions))
    emotion_color_map = {emotion: color for emotion,
                         color in zip(emotions, colors)}

    for _, emotion in enumerate(emotions):
        neighborhood_path = f'output/{emotion}/neighborhood.json'
        with open(neighborhood_path, 'r') as file:
            neighborhood_data = json.load(file)
        neighborhood_df = pd.DataFrame([entry['emotion'] for entry in neighborhood_data], columns=[
                                       f'emotion_{i}' for i in range(28)])
        neighborhood_embedding = reducer.fit_transform(neighborhood_df.values)

        neighborhood_centroid = np.mean(neighborhood_embedding, axis=0)

        plt.scatter(neighborhood_centroid[0], neighborhood_centroid[1], color=emotion_color_map[emotion],
                    edgecolors='black', s=200, marker='o', label=f"{emotion} neighborhood centroid")

        for gen in range(1, 4):
            sentences_path = f'output/{emotion}/all_sentences_{gen}.json'
            with open(sentences_path, 'r') as file:
                sentences_data = json.load(file)
            sentences_df = pd.DataFrame([entry['emotion'] for entry in sentences_data], columns=[
                                        f'emotion_{i}' for i in range(28)])
            sentences_embedding = reducer.transform(sentences_df.values)
            sentences_centroid = np.mean(sentences_embedding, axis=0)

            plt.scatter(sentences_centroid[0], sentences_centroid[1], color=emotion_color_map[emotion],
                        edgecolors='red', s=150, marker='s', label=f"{emotion} model gen {gen} centroid")
            plt.text(sentences_centroid[0], sentences_centroid[1], str(
                gen), color='black', fontsize=12, ha='center', va='center')

    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.title(title)
    plt.legend(title="Emotion", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


# plot_emotion_group_centroids(group_1, 'UMAP Projections by Emotion (Group 1)')
plot_emotion_group_centroids(group_2, 'UMAP Projections by Emotion (Group 2)')
# plot_emotion_group_centroids(group_3, 'UMAP Projections by Emotion (Group 3)')


# # CODE computes centroids first and then UMAP
# # Load data
# directory_path = 'output'
# emotions = [d for d in os.listdir(directory_path) if os.path.isdir(
#     os.path.join(directory_path, d))]

# group_1 = emotions[:8]
# group_2 = emotions[10:20]
# group_3 = emotions[20:]


# def plot_emotion_group_centroids(emotions, title):
#     reducer = umap.UMAP(n_neighbors=15, min_dist=0.1,
#                         n_components=2, metric='euclidean')
#     plt.figure(figsize=(12, 10))
#     sns.set_theme(style="whitegrid")

#     colors = sns.color_palette("hsv", len(emotions))
#     emotion_color_map = {emotion: color for emotion,
#                          color in zip(emotions, colors)}

#     all_centroids = []

#     for emotion in emotions:
#         neighborhood_path = f'{directory_path}/{emotion}/neighborhood.json'
#         with open(neighborhood_path, 'r') as file:
#             neighborhood_data = json.load(file)
#         neighborhood_df = pd.DataFrame([entry['emotion'] for entry in neighborhood_data], columns=[
#                                        f'emotion_{i}' for i in range(28)])
#         neighborhood_centroid = neighborhood_df.mean(axis=0)
#         all_centroids.append((emotion, 'neighborhood', neighborhood_centroid))

#         for gen in range(1, 4):
#             sentences_path = f'{directory_path}/{emotion}/all_sentences_{gen}.json'
#             with open(sentences_path, 'r') as file:
#                 sentences_data = json.load(file)
#             sentences_df = pd.DataFrame([entry['emotion'] for entry in sentences_data], columns=[
#                                         f'emotion_{i}' for i in range(28)])
#             sentences_centroid = sentences_df.mean(axis=0)
#             all_centroids.append(
#                 (emotion, f'model_gen_{gen}', sentences_centroid))

#     centroids_df = pd.DataFrame([centroid for _, _, centroid in all_centroids], index=[
#                                 (emotion, gen) for emotion, gen, _ in all_centroids])

#     centroids_embedding = reducer.fit_transform(centroids_df.values)

#     embedding_dict = {(emotion, gen): centroid for (
#         emotion, gen), centroid in zip(centroids_df.index, centroids_embedding)}

#     for (emotion, gen), centroid in zip(centroids_df.index, centroids_embedding):
#         label = f"{emotion} {gen}"
#         plt.scatter(centroid[0], centroid[1], color=emotion_color_map[emotion],
#                     edgecolors='black', s=200 if 'neighborhood' in gen else 150,
#                     marker='o' if 'neighborhood' in gen else 's', label=label)
#         plt.text(centroid[0], centroid[1], gen.replace('model_gen_', '') if 'gen' in gen else '',
#                  color='black', fontsize=12, ha='center', va='center')

#     for emotion in emotions:
#         coords = [embedding_dict[(emotion, gen)] for gen in [
#             'model_gen_1', 'model_gen_2', 'model_gen_3', 'neighborhood']]
#         coords = np.array(coords)
#         plt.plot(coords[:, 0], coords[:, 1], color=emotion_color_map[emotion],
#                  linestyle='-', linewidth=1, marker='o')

#     plt.xlabel('UMAP Dimension 1')
#     plt.ylabel('UMAP Dimension 2')
#     plt.title(title)
#     plt.legend(title="Emotion", bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.tight_layout()
#     plt.show()

# # plot_emotion_group_centroids(group_1, 'UMAP Projections by Emotion (Group 1)')
# # plot_emotion_group_centroids(group_2, 'UMAP Projections by Emotion (Group 2)')
# plot_emotion_group_centroids(group_3, 'UMAP Projections by Emotion (Group 3)')
