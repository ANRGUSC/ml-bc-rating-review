import json
import csv
import os
import logging
import pathlib 
from services.utilities import calculatePercentageSimilarity
from services.spotify_service import get_track, get_song_attributes, get_song_recommendations
from services.user import User

thisdir = pathlib.Path(__file__).parent.absolute()

def main():
    logging.basicConfig(level=logging.INFO)

    num_users = 3
    num_recs = 5
    results = "./results"
    users = [User.random() for _ in range(num_users)]
    data = []

    if not os.path.exists(results):
        os.makedirs(results)
    else: 
        for file_name in os.listdir(results):
            file_path = os.path.join(results, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)

    for user in users:
        recommendations = []

        for track_id in user.get_recommendations(num_recs):
            track = get_track(track_id)
            attributes = get_song_attributes(track_id)
            recommendations.append({
                "track_id": track_id,
                "title": track["name"],
                "artists": track["artists"],
                "attributes": attributes,
            })

            similarity = calculatePercentageSimilarity(attributes, user.attributes)
            similarity["song_name"] = track["name"]
            similarity["song_artist"] = track["artists"][0]["name"]

            column_order = ["song_name", "song_artist"] + list(user.attributes.keys()) + ["average"]

            with open(f"{results}/{user.user_id}.csv", "a") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=column_order)

                if csvfile.tell() == 0:
                    writer.writeheader()

                writer.writerow(similarity)
       
        data.append({
            "user_id": user.user_id,
            "attributes": user.attributes,
            "recommendations": recommendations,
        })

    thisdir.joinpath("gpt-recommendations.json").write_text(json.dumps(data, indent=2)) 

if __name__ == "__main__":
    main()