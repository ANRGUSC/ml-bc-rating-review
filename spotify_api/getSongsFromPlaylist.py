from dotenv import load_dotenv
import os
import csv
    
from services import authentication_service, playlist_service, song_service
from models.models import SongWithRecommendations


load_dotenv()

access_token = authentication_service.get_auth_token(client_id=os.getenv('CLIENT_ID'), client_secret=os.getenv('CLIENT_SECRET'))

songsWithRecommendations = {}


if access_token:
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    playlist_id = "37i9dQZF1EIeuoqE4dmQIB"
    songs = playlist_service.get_songs_from_playlist(playlist_id, headers)

    for song in songs:
        newFeature = song_service.get_song_features(song.song_id, headers)
        recommendations = song_service.get_song_recommendations(song.song_id, newFeature, headers)

        songWithRecommendations = SongWithRecommendations(song, newFeature, recommendations)
        songsWithRecommendations[song.song_name] = songWithRecommendations

else:
    print("Failed to authenticate.\n")
    exit()


# for song in songsWithRecommendations:
#     print(song)

### TODO: make this a function from a diff
# with open("prompts.txt", "w") as file:
#     for song_name, songWithRecommendations in songsWithRecommendations.items():
#         file.write(f"You are a music expert Whose job is to look at a song name and a list of the song attributes and produce recommended songs based off of the provided song and attribute values.\n")
#         file.write(f"The current song is {songWithRecommendations.song.song_name} by {songWithRecommendations.song.song_artist} and the attributes for the song are as follows: {songWithRecommendations.features}.\n")
#         file.write("Acousticness describes how acoustic a song is. A score of 1.0 means the song is most likely to be an acoustic one.\n")
#         file.write("Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.\n")
#         file.write("Energy represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy.\n")
#         file.write("The instrumentalness value represents the amount of vocals in the song. The closer it is to 1.0, the more instrumental the song is.\n")
#         file.write("Liveness, when above 0.8 provides strong likelihood that the track is live.\n")
#         file.write("The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typically range between -60 and 0 db.\n")
#         file.write("Speechiness detects the presence of spoken words in a track. If the speechiness of a song is above 0.66, it is probably made of spoken words, a score between 0.33 and 0.66 is a song that may contain both music and words, and a score below 0.33 means the song does not have any speech.\n")
#         file.write("Tempo represents the overall estimated tempo of a track in beats per minute (BPM).\n")
#         file.write("The time signature (meter) is a notational convention to specify how many beats are in each bar (or measure). The time signature ranges from 3 to 7 indicating time signatures of 3/4, to 7/4.\n")
#         file.write("Valence is a measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative.\n")
#         file.write("Please create a list of 10 recommended songs based on these attributes.\n")
#         file.write("\n\n\n") 



######### THIS IS JUST TO GET A SPECIFIC SONG FROM THE PLAYLIST'S RECOMMENDED SONGS
# rec_songs = songsWithRecommendations['I\'m Your Puppet'].recommended_songs


# for song in rec_songs:
#     print(f"Song ID: {song.song_id}")
#     print(f"Song Name: {song.song_name}")
#     print(f"Song Artist: {song.song_artist}")
#     print("\n")






