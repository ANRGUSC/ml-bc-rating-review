import json
import config
import numpy as np
import pathlib
from openai import OpenAI
import sys
from dotenv import load_dotenv

load_dotenv()

thisdir = pathlib.Path(__file__).parent.absolute()

def main(emotion_target):
    outputdir = thisdir.joinpath(f"output/{emotion_target}")
    outputdir.mkdir(parents=True, exist_ok=True)

    sentences = json.loads(thisdir.joinpath("model_outputs.json").read_text())

    label_order = {emotion_target: 0}
    for i, emotion in enumerate(sentences[0]["emotion"], start=1):
        if emotion["label"] not in label_order:
            label_order[emotion["label"]] = i

    settings_path = outputdir.joinpath('experiment_settings.json')
    with open(settings_path, 'w') as file:
        json.dump({'emotion_target': emotion_target,
                  'label_order': label_order}, file, indent=4)

    # convert emotion into a numpy array in the order of the label_order
    sentence_arrays = [
        {
            "text": sentence["text"],
            "emotion": np.array([emotion["score"] for emotion in sorted(sentence["emotion"], key=lambda x: label_order[x["label"]])])
        }
        for sentence in sentences
    ]

    sorted_sentence_arrays = sorted(
        sentence_arrays, key=lambda x: x["emotion"][0], reverse=True)
    
    sentence = sorted_sentence_arrays[0]

    # get k nearest neighbors to the sentence
    k = 100
    # get the distances between the sentence and all other sentences
    distances = np.array([np.linalg.norm(
        sentence["emotion"] - other_sentence["emotion"]) for other_sentence in sentence_arrays])
    # get the indices of the k nearest neighbors
    neighborhood = np.argsort(distances)[:k+1]

    neighborhood_file = outputdir.joinpath("neighborhood.json")
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
                    {"role": "user", "content": config.prompt},
                    {"role": "assistant",
                        "content": sentence_arrays[i]["text"]}
                ]
            })
            for i in nearest_neighbors[:k_fraction]
        ]

        fine_tuning_file = outputdir.joinpath(f"fine_tuning_{k_fraction}.jsonl")
        fine_tuning_file.write_text("\n".join(lines), encoding="utf-8")

        res = config.client.files.create(
            file=fine_tuning_file.open("rb"),
            purpose="fine-tune"
        )

        res = config.client.fine_tuning.jobs.create(
            training_file=res.id,
            model="gpt-3.5-turbo"
        )

        print(f"Fine-tuning with {k_fraction} nearest neighbors")
        print(res)

        fine_tune_jobs[k_fraction] = res.id

    fine_tune_jobs_file = outputdir.joinpath("fine_tune_jobs.json")
    fine_tune_jobs_file.write_text(json.dumps(
        fine_tune_jobs, indent=4), encoding="utf-8")


if __name__ == "__main__":
    if (len(sys.argv) > 1):
         emotion_target = sys.argv[1]
    main(emotion_target)
