import pandas as pd
import pathlib
import json
from transformers import pipeline


thisdir = pathlib.Path(__file__).parent.absolute()
classifier = pipeline(task="text-classification",
                      model="SamLowe/roberta-base-go_emotions", top_k=None, device="mps")


def main():
    # sentences = [
    #     "I am not having a great day",
    #     "I am having a great day",
    #     "I like apples",
    #     "I hate apples",
    #     "I am feeling anxious",
    #     "I am feeling excited",
    # ]

    # load sentences from .parquet file
    df = pd.read_parquet(thisdir.joinpath("sentences.parquet"))
    sentences = df["text"].tolist()

    model_outputs = classifier(sentences)

    json_output = [
        {
            "text": sentence,
            "emotion": model_output
        }
        for sentence, model_output in zip(sentences, model_outputs)
    ]

    thisdir.joinpath("model_outputs.json").write_text(
        json.dumps(json_output, indent=4), encoding="utf-8")


if __name__ == "__main__":
    main()
