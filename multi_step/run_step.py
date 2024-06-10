import pathlib
import json
import random
from typing import List
import numpy as np
from openai import OpenAI
import os
import dotenv

from fine_tune_initial import prompt, sentence_arrays, label_order
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
    num_sentences = 50

    model_gen = 3

    # load fine_tune_jobs from .json file
    fine_tune_jobs = json.loads(thisdir.joinpath("fine_tune_jobs_3.json").read_text())
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    #read the previous all_sentences.json file
    all_sentences = json.loads(thisdir.joinpath("all_sentences.json").read_text())

    models = {}
    for k_fraction, fine_tune_job_id in fine_tune_jobs.items():
        # get trained model
        res = client.fine_tuning.jobs.retrieve(
            fine_tuning_job_id=fine_tune_job_id
        )

        models[int(k_fraction)] = res.fine_tuned_model

    neighborhood = json.loads(thisdir.joinpath("neighborhood.json").read_text())
    sentence = neighborhood[0]
    all_sentences = []
    local_distance_array = []
    global_distance_array = []
    for k_fraction, fine_tuned_model in models.items():
        sentences = []
        for i in range(num_sentences):
            print(f"Generating sentence {i+1}/{num_sentences} for model {k_fraction}")
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
        #Calculate the distance between the sentence's emotion and the emotions of the generated sentences
        for i in range(num_sentences):
            distance = np.linalg.norm(sentence["emotion"] - np.array(emotions[i]["emotion"]))
            local_distance_array.append(distance)

        average_distance = np.mean(local_distance_array)
        global_distance_array.append(average_distance)

                            
        all_sentences.append([
            {
                "k_fraction": k_fraction,
                "model_gen": model_gen,
                "text": emotion["text"],
                "type": "output-ft",
                "emotion": emotion["emotion"].tolist(),
            }
            for emotion in emotions
        ])

    thisdir.joinpath("all_sentences_2.json").write_text(json.dumps(all_sentences, indent=4), encoding="utf-8")
    print(global_distance_array)
    # choose the model with the smallest distance
    min_distance = np.argmin(global_distance_array)
    print(min_distance)
    #Get best model's fine-tuning id
    best_model = models[(min_distance+1)*33]
    print(f"Best model: {best_model}")
    #Get best model's fine-tuning job id
    best_model_id = fine_tune_jobs[str((min_distance+1)*33)]
    print(f"Best model id: {best_model_id}")
    #save best model's fine-tuning job id

    thisdir.joinpath("best_model_id.json").write_text(json.dumps(best_model, indent=4), encoding="utf-8")


    # Choose the model with the best performance
    # k_fractions = [int(k*1/3), int(k*2/3), k]

    # sentences = []
    # for i in range(num_sentences):
    #     try:
    #         examples = []
    #         for example in random.choices(neighborhood, k=10):
    #             examples.append({"role": "user", "content": prompt})
    #             examples.append({"role": "assistant", "content": example["text"]})
    #         res = client.chat.completions.create(
    #             model='gpt-3.5-turbo',
    #             messages=[
    #                 *examples,
    #                 {"role": "user", "content": prompt}
    #             ],
    #         )
    #         response = res.choices[0].message.content
    #         sentences.append(response)
    #     except Exception as e:
    #         print(e)
    #         print(f"Error generating sentence {i+1}/{num_sentences} for k_fraction {k_fraction}\n\tmodel={fine_tuned_model}\n\prompt={prompt}")
    #         continue
    # emotions = get_embedding(sentences)
    # all_sentences.extend([
    #     {
    #         "k_fraction": k_fraction,
    #         "text": emotion["text"],
    #         "type": "output-pe",
    #         "emotion": emotion["emotion"].tolist(),
    #     }
    #     for emotion in emotions
    # ])

    


if __name__ == "__main__":
    main()