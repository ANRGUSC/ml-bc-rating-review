import pathlib
import json
from typing import List
import numpy as np
from openai import OpenAI
import os
import dotenv
import pandas as pd
import plotly.express as px

from fine_tune import prompt, sentence_arrays, label_order
# from prepare import classifier

dotenv.load_dotenv()

thisdir = pathlib.Path(__file__).parent.absolute()

def main():
    neighborhood = json.loads(thisdir.joinpath("neighborhood.json").read_text())
    output_sentences = json.loads(thisdir.joinpath("all_sentences.json").read_text())

    # target_sentence = neighborhood[0]
    # get center of neighborhood
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
    savepath = thisdir.joinpath("output/distance.html")
    savepath.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(savepath)

if __name__ == "__main__":
    main()    
