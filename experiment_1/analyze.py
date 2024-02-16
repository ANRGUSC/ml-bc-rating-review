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

    target_sentence = neighborhood[0]

    rows = []
    for i, sentence in enumerate(neighborhood[1:], start=1):
        distance = np.linalg.norm(np.array(target_sentence["emotion"]) - np.array(sentence["emotion"]))
        rows.append([i, "neighborhood", None, distance])

    for i, sentence in enumerate(output_sentences, start=1):
        distance = np.linalg.norm(np.array(target_sentence["emotion"]) - np.array(sentence["emotion"]))
        rows.append([i, "output", sentence["k_fraction"], distance])

    df = pd.DataFrame(rows, columns=["num", "type", "k_fraction", "distance"])
    print(df)
    fig = px.box(
        df,
        x="k_fraction",
        y="distance",
        color="type",
        title="Distance from target sentence",
        points="all",
        # category_orders={"k_fraction": label_order},
        template="plotly_white",
    )
    savepath = thisdir.joinpath("output/distance.html")
    savepath.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(savepath)

if __name__ == "__main__":
    main()    
