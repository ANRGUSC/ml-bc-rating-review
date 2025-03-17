import pathlib
import json
import numpy as np
import dotenv
import sys
import config
from typing import List
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


def main(emotion_target, output_name):
    outputdir = thisdir.joinpath(f"{output_name}/{emotion_target}")
    fine_tuning_dir = outputdir.joinpath("fine_tuning_files")
    settings_path = outputdir.joinpath('experiment_settings.json')

    with open(settings_path, 'r') as file:
        settings = json.load(file)

    label_order = settings['label_order']

    num_sentences = 50
    model_gen = 1

    # load fine_tune_jobs from .json file
    fine_tune_jobs_file = fine_tuning_dir.joinpath("fine_tune_jobs_initial.json")
    fine_tune_jobs = json.loads(fine_tune_jobs_file.read_text())

    models = {}
    for key_str, fine_tune_job_id in fine_tune_jobs.items():
        # get trained model
        res = config.client.fine_tuning.jobs.retrieve(
            fine_tuning_job_id=fine_tune_job_id
        )

        models[int(key_str)] = res.fine_tuned_model

    neighborhood = json.loads(outputdir.joinpath("neighborhood.json").read_text())
    sentence = neighborhood[0]

    all_sentences = []
    global_distance_array = []

    sorted_sub_model_keys = sorted(models.keys())

    for sub_model_idx in sorted_sub_model_keys:
        fine_tuned_model = models[sub_model_idx]
        sentences = []

        for i in range(num_sentences):
            print(
                f"Generating sentence {i+1}/{num_sentences} for sub model {sub_model_idx}")
            try:
                res = config.client.chat.completions.create(
                    model=fine_tuned_model,
                    messages=[{"role": "user", "content": config.prompt}],
                )
            except Exception as e:
                print(e)
                print(
                    f"Error generating sentence {i+1}/{num_sentences} for k_fraction {sub_model_idx}\n\tmodel={fine_tuned_model}\n\prompt={config.prompt}")
                continue
            response = res.choices[0].message.content
            sentences.append(response)

        emotions = get_embedding(sentences, label_order)

        # Calculate the distance between the sentence's emotion and the emotions of the generated sentences
        local_distance_array = []
        for i in range(num_sentences):
            distance = np.linalg.norm(
                sentence["emotion"] - np.array(emotions[i]["emotion"]))
            local_distance_array.append(distance)

        average_distance = np.mean(local_distance_array)
        global_distance_array.append(average_distance)

        all_sentences.extend([
            {
                "sub_model_idx": sub_model_idx,
                "model_gen": model_gen,
                "text": emotion["text"],
                "type": "output-ft",
                "emotion": emotion["emotion"].tolist(),
            }
            for emotion in emotions
        ])
 
    print("Average distances:", global_distance_array)
    # choose the model with the smallest distance
    min_distance = np.argmin(global_distance_array)
    best_sub_model_key = sorted_sub_model_keys[min_distance]
    print(f"Best sub-model key: {best_sub_model_key}")

    # Get best model's fine-tuning id
    best_model = models[best_sub_model_key]
    print(f"Best model (name): {best_model}")

    # Get best model's fine-tuning job id
    best_model_id = fine_tune_jobs[str(best_sub_model_key)]
    print(f"Best model job ID: {best_model_id}")
    
    # save best model's fine-tuning job id
    fine_tuning_dir.joinpath("best_model_id_1.json").write_text(
        json.dumps(best_model, indent=4), encoding="utf-8")

    best_sentences = [
        sentence for sentence in all_sentences if sentence['sub_model_idx'] == best_sub_model_key]

    # only write best sentences to all_sentences_1.json
    outputdir.joinpath("all_sentences_1.json").write_text(
        json.dumps(best_sentences, indent=4), encoding="utf-8")
    
    settings[f"best_model_{model_gen}"] = best_sub_model_key

    # Save updated settings back to experiment_settings.json
    with open(settings_path, 'w') as file:
        json.dump(settings, file, indent=4)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        emotion_target = sys.argv[1]
        output_name = sys.argv[2]
    main(emotion_target, output_name)
