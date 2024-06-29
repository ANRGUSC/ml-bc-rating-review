import json
import random
import sys
import pathlib
import config
from dotenv import load_dotenv

load_dotenv()

thisdir = pathlib.Path(__file__).parent.absolute()
settings_path = thisdir.joinpath('experiment_settings.json')

with open(settings_path, 'r') as file:
    settings = json.load(file)

emotion_target = settings['emotion_target']

sentences = json.loads(thisdir.joinpath("model_outputs.json").read_text())
best_model = json.loads(thisdir.joinpath("best_model_id.json").read_text())


def main(model_gen):
    neighborhood = json.loads(
        thisdir.joinpath(f"output/{emotion_target}/neighborhood.json").read_text())

    # get k nearest neighbors to the sentence
    k = 100

    # shuffle the neighborhood
    random.shuffle(neighborhood)
    nearest_neighbors = neighborhood

    k_fractions = [int(k*1/3), int(k*1/3), int(k*1/3)]
    k_fractions = [f for f in k_fractions if f >= 10]
    fine_tune_jobs = {}
    current_line = 0
    for k_fraction in k_fractions:
        lines = [
            json.dumps({
                "messages": [
                    {"role": "user", "content": config.prompt},
                    {"role": "assistant",
                        "content": neighbor["text"]}
                ]
            })
            for neighbor in nearest_neighbors[current_line:current_line+k_fraction]
        ]
        current_line += k_fraction

        fine_tuning_file = thisdir.joinpath(
            "fine_tuning_files", f"model_1_{current_line/33}.jsonl")
        fine_tuning_file.parent.mkdir(exist_ok=True)
        fine_tuning_file.write_text("\n".join(lines), encoding="utf-8")

        res = config.client.files.create(
            file=fine_tuning_file.open("rb"),
            purpose="fine-tune"
        )

        res = config.client.fine_tuning.jobs.create(
            training_file=res.id,
            model=best_model
        )

        print(f"Fine-tuning with model {current_line/33}")
        print(res)

        fine_tune_jobs[current_line] = res.id

    fine_tune_jobs_file = thisdir.joinpath(f"fine_tune_jobs_{model_gen}.json")
    fine_tune_jobs_file.write_text(json.dumps(
        fine_tune_jobs, indent=4), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_gen = sys.argv[1]
    main(model_gen)
