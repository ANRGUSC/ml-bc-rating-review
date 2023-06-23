import requests
from models.models import Song

def get_songs_from_playlist(playlist_id, headers):
    songsResponse = requests.get(
        f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks?limit=50", headers=headers)

    if songsResponse.status_code == 200:
        playlist_items = songsResponse.json()["items"]
        songs = []

        for item in playlist_items:
            newSong = Song(item['track']['id'], item['track']
                           ['name'], item['track']['artists'][0]['name'])
            songs.append(newSong)

        return songs
    else:
        return None
