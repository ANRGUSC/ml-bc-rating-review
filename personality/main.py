import json
import logging
import random
import uuid
import requests
from typing import Dict, List
import dotenv
import os
from functools import lru_cache
import openai
import pathlib 

thisdir = pathlib.Path(__file__).parent.absolute()

dotenv.load_dotenv()

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
openai.api_key = os.getenv("OPENAI_API_KEY")

@lru_cache(maxsize=1)
def get_access_token() -> str:
    """Gets an access token for the Spotify API"""
    # curl -X POST "https://accounts.spotify.com/api/token" \
    #  -H "Content-Type: application/x-www-form-urlencoded" \
    #  -d "grant_type=client_credentials&client_id=your-client-id&client_secret=your-client-secret"

    res = requests.post(
        "https://accounts.spotify.com/api/token",
        data={
            "grant_type": "client_credentials",
            "client_id": SPOTIFY_CLIENT_ID,
            "client_secret": SPOTIFY_CLIENT_SECRET,
        },
    )
    res.raise_for_status()
    return res.json()["access_token"]

def spotify_find_track(title: str, artists: List[str]) -> str:
    """Finds the Spotify ID of a track given its title and artists"""
    res = requests.get(
        "https://api.spotify.com/v1/search",
        params={
            "q": f"{title} {' '.join(artists)}",
            "type": "track",
            "limit": 1,
        },
        headers={
            "Authorization": f"Bearer {get_access_token()}",
        },
    )
    tracks = res.json()
    if len(tracks["tracks"]["items"]) == 0:
        raise ValueError(f"No track found for {title} by {artists}")
    return tracks["tracks"]["items"][0]["id"]

def get_track(track_id: str) -> Dict:
    res = requests.get(
        f"https://api.spotify.com/v1/tracks/{track_id}",
        params={
            "ids": track_id,
        },
        headers={
            "Authorization": f"Bearer {get_access_token()}",
        }
    )
    res.raise_for_status()
    return res.json()


ATTRIBUTES = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness', 'valence', 'popularity']
def get_random_song_attributes() -> Dict[str, float]:
    """Gets a random set of song attributes"""
    attributes = {attr: round(random.random(), 3) for attr in ATTRIBUTES}
    attributes["popularity"] = random.randint(0, 100)
    return attributes

def get_song_attributes(track_id: str) -> List[float]:
    res = requests.get(
        f"https://api.spotify.com/v1/audio-features/{track_id}",
        params={
            "ids": track_id,
        },
        headers={
            "Authorization": f"Bearer {get_access_token()}",
        }
    )
    res.raise_for_status()

    attributes = {attr: value for attr, value in res.json().items() if attr in ATTRIBUTES}

    # get popularity
    res = requests.get(
        f"https://api.spotify.com/v1/tracks/{track_id}",
        params={
            "ids": track_id,
        },
        headers={
            "Authorization": f"Bearer {get_access_token()}",
        }
    )
    res.raise_for_status()
    attributes["popularity"] = res.json()["popularity"]

    return attributes

class User:
    def __init__(self, user_id: str, attributes: Dict[str, float]):
        self.user_id = user_id
        self.attributes = attributes

    @classmethod
    def random(cls):
        return cls(user_id=uuid.uuid4().hex, attributes=get_random_song_attributes())
    
    def get_recommendations(self, num: int):
        functions = [
            {
                "name": "spotify_find_track",
                "description": "Finds the Spotify ID of a track given its title and artists",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "artists": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["title", "artists"],
                }
            }
        ]

        recs = []
        recs_history = []
        while len(recs) < num:
            conversation = [
                {"role": "system", "content": f"You are a music enthusiast with the following taste profile: {self.attributes}. Your job is to give song recommendations (spotify track IDs)."},
                {"role": "user", "content": "Give me a song recommendation. You have already recommended the following songs: " + ", ".join(recs_history)},
            ]
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=conversation,
                functions=functions,
                function_call={"name": "spotify_find_track"},  # auto is default, but we'll be explicit
            )
            response_message = response["choices"][0]["message"]

            function_args = json.loads(response_message["function_call"]["arguments"])
            try:
                track_id = spotify_find_track(function_args["title"], function_args["artists"])
                logging.info(f"Recommended {track_id} ({function_args['title']}, {function_args['artists']})")
                # track = get_track(track_id)
                # attributes = get_song_attributes(track_id)
            except Exception as e:
                logging.error(f"Error getting track: {function_args['title'], function_args['artists']}\n{e}")

            if track_id in recs:
                logging.info(f"Already recommended {track_id}")
                continue

            recs.append(track_id)
            recs_history.append(f"{function_args['title']} {function_args['artists']}")

        return recs
    
def main():
    logging.basicConfig(level=logging.INFO)

    num_users = 3
    num_recs = 5
    users = [User.random() for _ in range(num_users)]
    data = []
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

        data.append({
            "user_id": user.user_id,
            "attributes": user.attributes,
            "recommendations": recommendations,
        })

    thisdir.joinpath("data.json").write_text(json.dumps(data, indent=2))
        
        




if __name__ == "__main__":
    main()