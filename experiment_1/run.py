import pathlib
import json
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

    print(models)
    num_sentences = 5
    all_sentences = []
    for k_fraction, fine_tuned_model in models.items():
        sentences = []
        for i in range(num_sentences):
            res = client.chat.completions.create(
                model=fine_tuned_model,
                messages=[{"role": "user", "content": prompt}],
            )
            
            response = res.choices[0].message.content
            sentences.append(response)
        
        emotions = get_embedding(sentences)
        all_sentences.extend([
            {
                "k_fraction": k_fraction,
                "text": emotion["text"],
                "emotion": emotion["emotion"].tolist()
            }
            for emotion in emotions
        ])

    thisdir.joinpath("all_sentences.json").write_text(json.dumps(all_sentences, indent=4), encoding="utf-8")


if __name__ == "__main__":
    main()