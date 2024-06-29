import pathlib
import json
import numpy as np
import sys
import dotenv
import config
from typing import List

from prepare import classifier

dotenv.load_dotenv()

thisdir = pathlib.Path(__file__).parent.absolute()
settings_path = thisdir.joinpath('experiment_settings.json')

with open(settings_path, 'r') as file:
    settings = json.load(file)

emotion_target = settings['emotion_target']
label_order = settings['label_order']


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


def main(model_gen):
    num_sentences = 50

    # load fine_tune_jobs from .json file
    fine_tune_jobs = json.loads(thisdir.joinpath(
        f"fine_tune_jobs_{model_gen}.json").read_text())

    # read the previous all_sentences.json file
    all_sentences = json.loads(thisdir.joinpath(
        f"output/{emotion_target}/all_sentences_{model_gen - 1}.json").read_text())

    models = {}
    for k_fraction, fine_tune_job_id in fine_tune_jobs.items():
        # get trained model
        res = config.client.fine_tuning.jobs.retrieve(
            fine_tuning_job_id=fine_tune_job_id
        )

        models[int(k_fraction)] = res.fine_tuned_model

    neighborhood = json.loads(thisdir.joinpath(
        f"output/{emotion_target}/neighborhood.json").read_text())
    sentence = neighborhood[0]
    all_sentences = []
    local_distance_array = []
    global_distance_array = []
    for k_fraction, fine_tuned_model in models.items():
        sentences = []
        for i in range(num_sentences):
            print(
                f"Generating sentence {i+1}/{num_sentences} for model {k_fraction}")
            try:
                res = config.client.chat.completions.create(
                    model=fine_tuned_model,
                    messages=[{"role": "user", "content": config.prompt}],
                )
            except Exception as e:
                print(e)
                print(
                    f"Error generating sentence {i+1}/{num_sentences} for k_fraction {k_fraction}\n\tmodel={fine_tuned_model}\n\prompt={config.prompt}")
                continue
            response = res.choices[0].message.content
            sentences.append(response)

        emotions = get_embedding(sentences)

        # Calculate the distance between the sentence's emotion and the emotions of the generated sentences
        for i in range(num_sentences):
            distance = np.linalg.norm(
                sentence["emotion"] - np.array(emotions[i]["emotion"]))
            local_distance_array.append(distance)

        average_distance = np.mean(local_distance_array)
        global_distance_array.append(average_distance)

        all_sentences.extend([
            {
                "k_fraction": k_fraction,
                "model_gen": model_gen,
                "text": emotion["text"],
                "type": "output-ft",
                "emotion": emotion["emotion"].tolist(),
            }
            for emotion in emotions
        ])

    print(global_distance_array)
    # choose the model with the smallest distance
    min_distance = np.argmin(global_distance_array)
    print(min_distance)
    # Get best model's fine-tuning id
    best_model = models[(min_distance+1)*33]
    print(f"Best model: {best_model}")
    # Get best model's fine-tuning job id
    best_model_id = fine_tune_jobs[str((min_distance+1)*33)]
    print(f"Best model id: {best_model_id}")
    # save best model's fine-tuning job id

    thisdir.joinpath("best_model_id.json").write_text(
        json.dumps(best_model, indent=4), encoding="utf-8")

    best_model_fraction = min_distance * 33 + 33
    best_sentences = [
        sentence for sentence in all_sentences if sentence['k_fraction'] == best_model_fraction]

    # only write the best sentences to all_sentences_{model_gen}.json
    thisdir.joinpath(f"output/{emotion_target}/all_sentences_{model_gen}.json").write_text(
        json.dumps(best_sentences, indent=4), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_gen = int(sys.argv[1])
    main(model_gen)
