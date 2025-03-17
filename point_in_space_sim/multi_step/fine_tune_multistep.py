import json
import sys
import pathlib
import config
import random
from dotenv import load_dotenv

load_dotenv()

thisdir = pathlib.Path(__file__).parent.absolute()

def main(emotion_target, model_gen):
    outputdir = thisdir.joinpath(f"output/{emotion_target}")
    
    fine_tuning_dir = outputdir.joinpath("fine_tuning_files")
    
    # 1) Read experiment settings to get emotion_target and the segments
    settings_path = outputdir.joinpath('experiment_settings.json')
    with settings_path.open('r', encoding="utf-8") as f:
        settings = json.load(f)

    best_model_index = settings[f'best_model_{model_gen - 1}']
    new_blocklist = settings[f'segments_{model_gen - 1}'][best_model_index - 1]
    old_blocklist = settings['blocklist']
    segments_to_exclude = new_blocklist + old_blocklist

    # 2) Read the best model from the previous round
    best_model_file = fine_tuning_dir.joinpath(f"best_model_id_{model_gen - 1}.json")
    best_model = json.loads(best_model_file.read_text())  # e.g. "ft:gpt-3.5-turbo-<job-id>"


    neighborhood_file = outputdir.joinpath("neighborhood.json")
    neighborhood = json.loads(neighborhood_file.read_text())

    nearest_neighbors = neighborhood[1:]
    filtered_nearest_neighbors = [neighbor for neighbor in nearest_neighbors if neighbor["index"] not in set(segments_to_exclude)]
    
    random.shuffle(filtered_nearest_neighbors)

    # 3) Split the neighborhood into 3 segments
    num_segments = 3
    k = len(filtered_nearest_neighbors)
    segment_size = k // num_segments
    segments = []
    for i in range(num_segments):
        start = i * segment_size
        end = start + segment_size
        segments.append(filtered_nearest_neighbors[start:end])
        
    segment_indices = [[n["index"] for n in seg] for seg in segments]

    # write segments to settings file
    settings[f'segments_{model_gen}'] = segment_indices
    with settings_path.open('w', encoding="utf-8") as f:
        json.dump(settings, f, indent=4)
    
    # 5) Fine-tune each sub-model (3 sub-models) using the selected segments
    fine_tune_jobs = {}
    for i in range(num_segments):
        # This is the list of original indices that belong to this segment
        curr_segment_idx = segment_indices[i]  # e.g. [42, 77, 123, ...]

        # Build training lines by filtering neighborhood for matching "index"
        lines = []
        for neighbor in neighborhood:
            if neighbor.get("index") in curr_segment_idx:
                # We found a neighbor that belongs to this segment
                lines.append(json.dumps({
                    "messages": [
                        {"role": "user", "content": config.prompt},
                        {"role": "assistant", "content": neighbor["text"]}
                    ]
                }))

        # Write the lines to a .jsonl file        
        fine_tuning_file = fine_tuning_dir.joinpath(f"model_{model_gen}_sub_{i + 1}.jsonl")
        fine_tuning_file.write_text("\n".join(lines), encoding="utf-8")

        # Upload the file
        res_file = config.client.files.create(
            file=fine_tuning_file.open("rb"),
            purpose="fine-tune"
        )

        # Create a fine-tuning job using the best_model
        res_ft = config.client.fine_tuning.jobs.create(
            training_file=res_file.id,
            model=best_model
        )

        print(f"[Round {model_gen}] Fine-tuning sub-model #{i} from base: {best_model}")
        print(res_ft)
        fine_tune_jobs[str(i + 1)] = res_ft.id

    # 6) Save the fine-tuning job IDs for reference
    fine_tune_jobs_file = fine_tuning_dir.joinpath(f"fine_tune_jobs_{model_gen}.json")
    fine_tune_jobs_file.write_text(
        json.dumps(fine_tune_jobs, indent=4),
        encoding="utf-8"
    )

    settings['blocklist'] = segments_to_exclude
    with settings_path.open('w', encoding="utf-8") as f:
        json.dump(settings, f, indent=4)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        emotion_target = sys.argv[1]
        model_gen = int(sys.argv[2])

    main(emotion_target, model_gen)
