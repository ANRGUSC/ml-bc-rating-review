import requests
from models.models import Song, Feature


def get_song_features(song_id, headers):
    featuresResponse = requests.get(
        f"https://api.spotify.com/v1/audio-features/{song_id}", headers=headers)

    if featuresResponse.status_code == 200:
        features = featuresResponse.json()

        newFeature = Feature(acousticness=features['acousticness'], danceability=features['danceability'],
                             energy=features['energy'], instrumentalness=features['instrumentalness'],
                             liveness=features['liveness'], loudness=features['loudness'],
                             speechiness=features['speechiness'], tempo=features['tempo'],
                             time_signature=features['time_signature'], valence=features['valence'])
        return newFeature
    else:
        return None


def get_song(song_id, headers):
    songResponse = requests.get(
        f"https://api.spotify.com/v1/tracks/{song_id}", headers=headers)

    if songResponse.status_code == 200:
        return songResponse.json()
    else: 
        return None
    
    
def get_songs(song_ids, headers):
    ids = ",".join(song_ids)
    songsResponse = requests.get(
    	f"https://api.spotify.com/v1/tracks", headers=headers, params={"ids": ids})
    
    if songsResponse.status_code == 200:
        return songsResponse.json()
    else:
        return None


def get_song_recommendations(song_id, feature, headers):
    params = {
        "limit": 20,
        "seed_tracks": song_id,
        "target_acousticness": feature.acousticness,
        "target_danceability": feature.danceability,
        "target_energy": feature.energy,
        "target_instrumentalness": feature.instrumentalness,
        "target_liveness": feature.liveness,
        "target_loudness": feature.loudness,
        "target_speechiness": feature.speechiness,
        "target_tempo": feature.tempo,
        "target_valence": feature.valence,
    }

    recommendedResponse = requests.get(
        f"https://api.spotify.com/v1/recommendations", headers=headers, params=params)

    if recommendedResponse.status_code == 200:
        recs = recommendedResponse.json()['tracks']
        recommendations = [Song(rec['id'], rec['name'], rec['artists'][0]['name']) for rec in recs]

        return recommendations
    else:
        return None
