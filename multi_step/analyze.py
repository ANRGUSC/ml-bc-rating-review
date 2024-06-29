import pathlib
import json
import numpy as np
import pandas as pd
import config
import plotly.express as px


thisdir = pathlib.Path(__file__).parent.absolute()
settings_path = thisdir.joinpath('experiment_settings.json')

with open(settings_path, 'r') as file:
    settings = json.load(file)

emotion_target = settings['emotion_target']


def main():
    neighborhood = json.loads(thisdir.joinpath(
        f"output/{emotion_target}/neighborhood.json").read_text())
    output_sentence_1 = json.loads(
        thisdir.joinpath(f"output/{emotion_target}/all_sentences_1.json").read_text())
    output_sentence_2 = json.loads(
        thisdir.joinpath(f"output/{emotion_target}/all_sentences_2.json").read_text())
    output_sentence_3 = json.loads(
        thisdir.joinpath(f"output/{emotion_target}/all_sentences_3.json").read_text())

    # get center of neighborhood
    center = np.mean([sentence["emotion"]
                     for sentence in neighborhood[1:]], axis=0)

    rows = []

    for i, sentence in enumerate(neighborhood[1:], start=1):
        distance = np.linalg.norm(center - np.array(sentence["emotion"]))
        rows.append([i, "neighborhood", "N/A", distance])

    for i, sentence in enumerate(output_sentence_1, start=1):
        distance = np.linalg.norm(center - np.array(sentence["emotion"]))
        label = f"model_gen={sentence['model_gen']}" if sentence["type"] == "output-ft" else "N/A"
        rows.append([i, sentence["type"], label, distance])

    for i, sentence in enumerate(output_sentence_2, start=1):
        distance = np.linalg.norm(center - np.array(sentence["emotion"]))
        label = f"model_gen={sentence['model_gen']}" if sentence["type"] == "output-ft" else "N/A"
        rows.append([i, sentence["type"], label, distance])

    for i, sentence in enumerate(output_sentence_3, start=1):
        distance = np.linalg.norm(center - np.array(sentence["emotion"]))
        label = f"model_gen={sentence['model_gen']}" if sentence["type"] == "output-ft" else "N/A"
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
        f"output/{emotion_target}/distance.html")
    savepath.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(savepath)


if __name__ == "__main__":
    main()
