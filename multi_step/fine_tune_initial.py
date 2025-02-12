import json
import random
import numpy as np
import pathlib
import sys
import config
from dotenv import load_dotenv

load_dotenv()

thisdir = pathlib.Path(__file__).parent.absolute()

def main(emotion_target):
    outputdir = thisdir.joinpath(f"output/{emotion_target}")
    outputdir.parent.mkdir(parents=True, exist_ok=True)
    settings_path = outputdir.joinpath('experiment_settings.json')
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    
    sentences_file = thisdir.joinpath("model_outputs.json")
    sentences = json.loads(sentences_file.read_text())

    # 2) Create a label_order so target emotion is first
    label_order = {emotion_target: 0}
    for i, emotion in enumerate(sentences[0]["emotion"], start=1):
        if emotion["label"] not in label_order:
            label_order[emotion["label"]] = i

    # 3) Convert each sentence to {text, emotion} (no index here yet)
    #    where the emotion array is sorted such that the target emotion is at index 0
    sentence_arrays = [
        {
            "text": s["text"],
            "emotion": np.array([
                em["score"]
                for em in sorted(s["emotion"], key=lambda x: label_order[x["label"]])
            ])
        }
        for s in sentences
    ]

    # 4) Identify the target sentence (highest score at emotion[0])
    sorted_sentence_arrays = sorted(
        sentence_arrays,
        key=lambda x: x["emotion"][0],
        reverse=True
    )
    target_sentence = sorted_sentence_arrays[0]

    # 6) Get the top (k+1) neighbors by distance, including the target
    k = 100
    distances = np.array([
        np.linalg.norm(target_sentence["emotion"] - s["emotion"])
        for s in sentence_arrays
    ])
    neighborhood_indices = np.argsort(distances)[:(k + 1)]  # indices of the 100 closest

    # 7) Build neighborhood_list with "index" referencing sentence_arrays
    #    so each neighbor dictionary knows which row it's from.
    #    Here we just reuse the local index in `sentence_arrays`,
    #    but if you wanted the "original" index from `enumerate(sentences)`,
    #    you'd have to store it earlier in sentence_arrays. For demonstration:
    neighborhood_list = [
        {
            "index": int(idx),  # attach the index from sentence_arrays (0..len-1)
            "text": sentence_arrays[idx]["text"],
            "emotion": sentence_arrays[idx]["emotion"].tolist()
        }
        for idx in neighborhood_indices
    ]
    
    out_file = outputdir.joinpath("neighborhood.json")
    out_file.write_text(
        json.dumps(neighborhood_list, indent=4),
        encoding="utf-8"
    )

    # The first item in this list is presumably the target (distance=0)
    neighborhood_list.pop(0)  # remove it so we don't reuse it

    # 8) Shuffle the remaining 99 neighbors
    random.shuffle(neighborhood_list)

    # 9) Split the neighborhood into 3 groups of floor(k/3) each
    num_segments = 3
    segment_size = k // num_segments
    segments = []
    for i in range(num_segments):
        start = i * segment_size
        end = start + segment_size
        segments.append(neighborhood_list[start:end])

    segments_by_index = [[n["index"] for n in seg] for seg in segments]

    settings_data = {
        "emotion_target": emotion_target,
        "label_order": label_order,
        "target_sample": {
            "text": target_sentence["text"],
            "emotion": target_sentence["emotion"].tolist()
        },
        "segments_1": segments_by_index,
        "blocklist": []
    }
    settings_path.write_text(json.dumps(settings_data, indent=4), encoding="utf-8")

    # 11) Fine-tune three sub-models, each on a different segment (e.g., the first 3)
    fine_tune_jobs = {}
    for sub_model_idx in range(3):
        sub_segment = segments[sub_model_idx]
        lines = []
        for neighbor in sub_segment:
            lines.append(json.dumps({
                "messages": [
                    {"role": "user", "content": config.prompt},
                    {"role": "assistant", "content": neighbor["text"]}
                ]
            }))

        # Write .jsonl
        fine_tuning_dir = outputdir.joinpath("fine_tuning_files")
        fine_tuning_dir.mkdir(parents=True, exist_ok=True)
        
        fine_tuning_file = fine_tuning_dir.joinpath(f"model_1_sub_{sub_model_idx + 1}.jsonl")
        fine_tuning_file.write_text("\n".join(lines), encoding="utf-8")

        # Upload + create fine-tuning job
        res_file = config.client.files.create(
            file=fine_tuning_file.open("rb"),
            purpose="fine-tune"
        )
        res_ft = config.client.fine_tuning.jobs.create(
            training_file=res_file.id,
            model="gpt-3.5-turbo"
        )
        print(f"Fine-tuning sub-model #{sub_model_idx}...")
        print(res_ft)
        fine_tune_jobs[str(sub_model_idx + 1)] = res_ft.id

    # 12) Save the fine-tune job IDs
    fine_tune_jobs_file = fine_tuning_dir.joinpath("fine_tune_jobs_initial.json")
    fine_tune_jobs_file.write_text(
        json.dumps(fine_tune_jobs, indent=4),
        encoding="utf-8"
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        emotion_target = sys.argv[1]

    main(emotion_target)
