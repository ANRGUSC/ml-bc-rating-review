import json
import random
import time
import sys
import numpy as np
import pathlib
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

thisdir = pathlib.Path(__file__).parent.absolute()
sentences = json.loads(thisdir.joinpath("model_outputs.json").read_text())

emotion_target = "love"

# Sets the label order such that emotion_target is first
label_order = {emotion_target: 0}
for i, emotion in enumerate(sentences[0]["emotion"], start=1):
    if emotion["label"] not in label_order:
        label_order[emotion["label"]] = i

# label_order = {emotion["label"]: i for i, emotion in enumerate(sentences[0]["emotion"])}

# convert emotion into a numpy array in the order of the label_order
# sentence_arrays = [
#     {
#         "text": sentence["text"],
#         "emotion": np.array([emotion["score"] for emotion in sentence["emotion"]])
#     }
#     for sentence in sentences
# ]

# sorts every sentence emotion array by label_order
sentence_arrays = [
    {
        "text": sentence["text"],
        "emotion": np.array([emotion["score"] for emotion in sorted(sentence["emotion"], key=lambda x: label_order[x["label"]])])
    }
    for sentence in sentences
]
prompt = "write a reddit comment."


def main():
    # sorts sentence arrays such that highest emotion_target is first in descending order
    sorted_sentence_arrays = sorted(
        sentence_arrays, key=lambda x: x["emotion"][0], reverse=True)

    # picks the highest emotion_target sentence
    sentence = sorted_sentence_arrays[0]
    # pick a random sentence
    # sentence = random.choice(sentence_arrays)

    # Dumps the chosen sentence into a JSON file
    sentence_file = thisdir.joinpath("sentence.json")
    sentence_file.write_text(
        json.dumps(
            {
                "text": sentence["text"],
                "emotion": sentence["emotion"].tolist()
            },
            indent=4
        ),
        encoding="utf-8"
    )

    # get k nearest neighbors to the sentence
    k = 100
    # get the distances between the sentence and all other sentences
    distances = np.array([np.linalg.norm(
        sentence["emotion"] - other_sentence["emotion"]) for other_sentence in sentence_arrays])
    # get the indices of the k nearest neighbors without sorting
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

    # randomize the neighborhood so that each k_fraction (model) gets varying degrees of emotion_target
    # otherwise k_fraction: 33 would have the highest emotion_target scores and 99, the lowest
    random.shuffle(neighborhood)

    nearest_neighbors = neighborhood[1:]

    print(f"Original sentence: {sentence['text']}")
    print("Nearest neighbors:")
    for i in nearest_neighbors:
        print(f"  {sentence_arrays[i]['text']}")

    # We're going to fine-tune our model using the k-nearest neighbors.
    # Our hypothesis is that the more nearest-neighbors we provide, the more likely
    # the model will produce outputs that are similar to chosen sentence.

    k_fractions = [int(k*1/3), int(k*1/3), int(k*1/3)]
    k_fractions = [f for f in k_fractions if f >= 10]
    fine_tune_jobs = {}
    current_line = 0

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
            for i in nearest_neighbors[current_line:current_line+k_fraction]
        ]
        current_line += k_fraction

        fine_tuning_file = thisdir.joinpath(
            "fine_tuning_files", f"model_1_{current_line/33}.jsonl")
        fine_tuning_file.parent.mkdir(exist_ok=True)
        fine_tuning_file.write_text("\n".join(lines), encoding="utf-8")

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        res = client.files.create(
            file=fine_tuning_file.open("rb"),
            purpose="fine-tune"
        )

        res = client.fine_tuning.jobs.create(
            training_file=res.id,
            model="gpt-3.5-turbo"
        )

        print(f"Fine-tuning with model {current_line/33}")
        print(res)

        fine_tune_jobs[current_line] = res.id

    fine_tune_jobs_file = thisdir.joinpath("fine_tune_jobs_initial.json")
    fine_tune_jobs_file.write_text(json.dumps(
        fine_tune_jobs, indent=4), encoding="utf-8")


if __name__ == "__main__":
    main()
