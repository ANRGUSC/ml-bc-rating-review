import pathlib
import json
import config
from typing import List
import numpy as np
from openai import OpenAI
import os
import dotenv
import sys

from prepare import classifier

dotenv.load_dotenv()

thisdir = pathlib.Path(__file__).parent.absolute()

def get_embedding(sentences: List[str], label_order):
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


def main(emotion_target):
    emotiondir = thisdir.joinpath(f"output/{emotion_target}")
    
    num_sentences = 50
    
    settings_path = emotiondir.joinpath('experiment_settings.json')
    with open(settings_path, 'r') as file:
        settings = json.load(file)
    
    label_order = settings['label_order']
        
    # load fine_tune_jobs from .json file
    fine_tune_jobs = json.loads(emotiondir.joinpath(
        "fine_tune_jobs.json").read_text())

    models = {}
    for k_fraction, fine_tune_job_id in fine_tune_jobs.items():
        # get trained model
        res = config.client.fine_tuning.jobs.retrieve(
            fine_tuning_job_id=fine_tune_job_id
        )

        models[int(k_fraction)] = res.fine_tuned_model

    all_sentences = []
    for k_fraction, fine_tuned_model in models.items():
        sentences = []
        for i in range(num_sentences):
            print(
                f"Generating sentence {i+1}/{num_sentences} for k_fraction {k_fraction}")
            try:
                res = config.client.chat.completions.create(
                    model=fine_tuned_model,
                    messages=[{"role": "user", "content": config.prompt}],
                )
            except Exception as e:
                print(e)
                print(
                    f"Error generating sentence {i+1}/{num_sentences} for k_fraction {k_fraction}\n\tmodel={fine_tuned_model}\n\prompt={prompt}")
                continue
            response = " ".join(res.choices[0].message.content.split()[:500])
            sentences.append(response)

        emotions = get_embedding(sentences, label_order)
        all_sentences.extend([
            {
                "k_fraction": k_fraction,
                "text": emotion["text"],
                "type": "output-ft",
                "emotion": emotion["emotion"].tolist(),
            }
            for emotion in emotions
        ])

    emotiondir.joinpath("all_sentences.json").write_text(
        json.dumps(all_sentences, indent=4), encoding="utf-8")

if __name__ == "__main__":
    if (len(sys.argv) > 1):
        emotion_target = sys.argv[1]
    main(emotion_target)
