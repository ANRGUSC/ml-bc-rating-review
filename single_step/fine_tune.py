import json
import random
import time
import numpy as np
import pathlib
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

thisdir = pathlib.Path(__file__).parent.absolute()
sentences = json.loads(thisdir.joinpath("model_outputs.json").read_text())

label_order = {emotion["label"]: i for i,
               emotion in enumerate(sentences[0]["emotion"])}
for sentence in sentences:
    sentence["emotion"] = sorted(
        sentence["emotion"], key=lambda x: label_order[x["label"]])

# convert emotion into a numpy array in the order of the label_order
sentence_arrays = [
    {
        "text": sentence["text"],
        "emotion": np.array([emotion["score"] for emotion in sentence["emotion"]])
    }
    for sentence in sentences
]
prompt = "write a reddit comment."


def main():
    # pick a random sentence
    sentence = random.choice(sentence_arrays)

    # get k nearest neighbors to the sentence
    k = 100
    # get the distances between the sentence and all other sentences
    distances = np.array([np.linalg.norm(
        sentence["emotion"] - other_sentence["emotion"]) for other_sentence in sentence_arrays])
    # get the indices of the k nearest neighbors
    neighborhood = np.argsort(distances)[:k+1]

    neighborhood_file = thisdir.joinpath("neighborhood.json")
    neighborhood_file.write_text(
        json.dumps(
            [
                {
                    "text": sentence_arrays[i]["text"],
                    "emotion": sentence_arrays[i]["emotion"].tolist(),
                }
                for i in neighborhood
            ],
            indent=4
        ),
        encoding="utf-8"
    )

    nearest_neighbors = neighborhood[1:]

    print(f"Original sentence: {sentence['text']}")
    print("Nearest neighbors:")
    for i in nearest_neighbors:
        print(f"  {sentence_arrays[i]['text']}")

    # We're going to fine-tune our model using the k-nearest neighbors.
    # Our hypothesis is that the more nearest-neighbors we provide, the more likely
    # the model will produce outputs that are similar to chosen sentence.

    k_fractions = [int(k*1/3), int(k*2/3), k]
    k_fractions = [f for f in k_fractions if f >= 10]
    fine_tune_jobs = {}
    for k_fraction in k_fractions:
        # create fine-tuning file
        # {"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, {"role": "user", "content": "What's the capital of France?"}, {"role": "assistant", "content": "Paris, as if everyone doesn't know that already."}]}
        lines = [
            json.dumps({
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant",
                        "content": sentence_arrays[i]["text"]}
                ]
            })
            for i in nearest_neighbors[:k_fraction]
        ]

        fine_tuning_file = thisdir.joinpath(
            "fine_tuning_files", f"{k_fraction}.jsonl")
        fine_tuning_file.parent.mkdir(exist_ok=True)
        fine_tuning_file.write_text("\n".join(lines), encoding="utf-8")

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY_KUBISHI"])
        res = client.files.create(
            file=fine_tuning_file.open("rb"),
            purpose="fine-tune"
        )

        res = client.fine_tuning.jobs.create(
            training_file=res.id,
            model="gpt-3.5-turbo"
        )

        print(f"Fine-tuning with {k_fraction} nearest neighbors")
        print(res)

        fine_tune_jobs[k_fraction] = res.id

    fine_tune_jobs_file = thisdir.joinpath("fine_tune_jobs.json")
    fine_tune_jobs_file.write_text(json.dumps(
        fine_tune_jobs, indent=4), encoding="utf-8")


if __name__ == "__main__":
    main()
