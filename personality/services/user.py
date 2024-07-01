import uuid
import json
import openai
import dotenv
import logging
import os
from typing import Dict
from services.spotify_service import spotify_find_track
from services.utilities import get_random_song_attributes, ATTRIBUTES_DESCRIPTION


dotenv.load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY_KUBISHI")


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
                {"role": "system",
                    "content": f"You are a music enthusiast with the following taste profile: {self.attributes}. {ATTRIBUTES_DESCRIPTION} Your job is to give song recommendations (spotify track IDs)."},
                {"role": "user", "content": "Give me a song recommendation. You have already recommended the following songs: " +
                    ", ".join(recs_history)},
            ]
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=conversation,
                functions=functions,
                # auto is default, but we'll be explicit
                function_call={"name": "spotify_find_track"},
            )
            response_message = response["choices"][0]["message"]

            function_args = json.loads(
                response_message["function_call"]["arguments"])
            try:
                track_id = spotify_find_track(
                    function_args["title"], function_args["artists"])
                logging.info(
                    f"Recommended {track_id} ({function_args['title']}, {function_args['artists']})")
                # track = get_track(track_id)
                # attributes = get_song_attributes(track_id)
            except Exception as e:
                logging.error(
                    f"Error getting track: {function_args['title'], function_args['artists']}\n{e}")

            if track_id in recs:
                logging.info(f"Already recommended {track_id}")
                continue

            recs.append(track_id)
            recs_history.append(
                f"{function_args['title']} {function_args['artists']}")

        return recs
