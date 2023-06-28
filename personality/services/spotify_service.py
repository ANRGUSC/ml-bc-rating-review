import requests
import dotenv
import os
from typing import Dict, List
from functools import lru_cache
from services.utilities import ATTRIBUTES

dotenv.load_dotenv()

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

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

    res.raise_for_status()

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

def get_song_recommendations(feature, user_id, num_recs):
    params = {
        "limit": num_recs,
        "target_acousticness": feature.acousticness,
        "target_danceability": feature.danceability,
        "target_energy": feature.energy,
        "target_instrumentalness": feature.instrumentalness,
        "target_liveness": feature.liveness,
        "target_speechiness": feature.speechiness,
        "target_popularity": feature.popularity,
        "target_valence": feature.valence,
    }

    recommendedResponse = requests.get(
        f"https://api.spotify.com/v1/recommendations", headers={"Authorization": f"Bearer {get_access_token()}",}, params=params)

    recommendedResponse.raise_for_status()
    tracks = recommendedResponse.json()['tracks']

    return {
        user_id: [{
            "track_id": track['id'],
            "title": track["name"],
            "artists": track['artists'],
            "attributes": {attr: track[attr] for attr in params.keys() if attr.startswith('target_')}
        } for track in tracks]
    }
