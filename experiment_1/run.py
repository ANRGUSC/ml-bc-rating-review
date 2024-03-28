import pathlib
import json
import random
from typing import List
import numpy as np
from openai import OpenAI
import os
import dotenv

from fine_tune import prompt, sentence_arrays, label_order
from prepare import classifier

dotenv.load_dotenv()

thisdir = pathlib.Path(__file__).parent.absolute()

def get_embedding(sentences: List[str]):
    model_outputs = classifier(sentences)
    model_outputs = [
        {
            "text": sentence,
            "emotion": sorted(model_output, key=lambda x: label_order[x["label"]])
        }
        for sentence, model_output in zip(sentences, model_outputs)
    ]

    # convert emotion into a numpy array in the order of the label_order
    sentence_arrays = [
        {
            "text": sentence["text"],
            "emotion": np.array([emotion["score"] for emotion in sentence["emotion"]])
        }
        for sentence in model_outputs
    ]

    return sentence_arrays

def main():
    num_sentences = 20

    # load fine_tune_jobs from .json file
    fine_tune_jobs = json.loads(thisdir.joinpath("fine_tune_jobs.json").read_text())
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    models = {0: "gpt-3.5-turbo"}
    for k_fraction, fine_tune_job_id in fine_tune_jobs.items():
        # get trained model
        res = client.fine_tuning.jobs.retrieve(
            fine_tuning_job_id=fine_tune_job_id
        )

        models[int(k_fraction)] = res.fine_tuned_model

    neighborhood = json.loads(thisdir.joinpath("neighborhood.json").read_text())
    all_sentences = []
    for k_fraction, fine_tuned_model in models.items():
        sentences = []
        for i in range(num_sentences):
            print(f"Generating sentence {i+1}/{num_sentences} for k_fraction {k_fraction}")
            try:
                res = client.chat.completions.create(
                    model=fine_tuned_model,
                    messages=[{"role": "user", "content": prompt}],
                )
            except Exception as e:
                print(e)
                print(f"Error generating sentence {i+1}/{num_sentences} for k_fraction {k_fraction}\n\tmodel={fine_tuned_model}\n\prompt={prompt}")
                continue
            response = res.choices[0].message.content
            sentences.append(response)

        emotions = get_embedding(sentences)
        all_sentences.extend([
            {
                "k_fraction": k_fraction,
                "text": emotion["text"],
                "type": "output-ft",
                "emotion": emotion["emotion"].tolist(),
            }
            for emotion in emotions
        ])

    sentences = []
    for i in range(num_sentences):
        try:
            examples = []
            for example in random.choices(neighborhood, k=10):
                examples.append({"role": "user", "content": prompt})
                examples.append({"role": "assistant", "content": example["text"]})
            res = client.chat.completions.create(
                model='gpt-3.5-turbo',
                messages=[
                    *examples,
                    {"role": "user", "content": prompt}
                ],
            )
            response = res.choices[0].message.content
            sentences.append(response)
        except Exception as e:
            print(e)
            print(f"Error generating sentence {i+1}/{num_sentences} for k_fraction {k_fraction}\n\tmodel={fine_tuned_model}\n\prompt={prompt}")
            continue
    emotions = get_embedding(sentences)
    all_sentences.extend([
        {
            "k_fraction": k_fraction,
            "text": emotion["text"],
            "type": "output-pe",
            "emotion": emotion["emotion"].tolist(),
        }
        for emotion in emotions
    ])

    thisdir.joinpath("all_sentences.json").write_text(json.dumps(all_sentences, indent=4), encoding="utf-8")


if __name__ == "__main__":
    main()