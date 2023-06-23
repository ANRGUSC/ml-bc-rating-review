class Song:
    def __init__(self, song_id, song_name, song_artist):
        self.song_id = song_id
        self.song_name = song_name
        self.song_artist = song_artist
        

# acousticness, danceability, energy, instrumentalness, liveness, loudness, speechiness, tempo, time_signature, valence
class Feature:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self):
        return str(self.__dict__)


class SongWithRecommendations:
    def __init__(self, song: Song, features: Feature, recommended_songs: list = None):
        self.song = song
        self.features = features
        self.recommended_songs = recommended_songs if recommended_songs is not None else []

    def __str__(self):
        recommended_songs_names = [song.song_name for song in self.recommended_songs]
        return f"Song: {self.song.song_name} by {self.song.song_artist}, Features: {str(self.features)}, Recommended Songs: {recommended_songs_names}"