# import sys
# from pathlib import Path

from dotenv import load_dotenv
import os
import csv
# root = str(Path(__file__).resolve().parent)
# if root not in sys.path:
#     sys.path.append(root)

from services import authentication_service, song_service
from models.models import Song, SongWithRecommendations

load_dotenv()

gptRecommendations = [
    "3SdTKo2uVsxFblQjpScoHy",
    "745H5CctFr12Mo7cqa1BMH",
    "7tqhbajSfrz2F7E1Z75ASX",
    "3Um9toULmYFGCpvaIPFw7l",
    "1OppEieGNdItZbE14gLBEv",
    "3zBhihYUHBmGd2bcQIobrF",
    "0KOE1hat4SIer491XKk4Pa",
    "0sUe1KY92PKd8tpkf7hvDa",
    "1tqT6DhmsrtQgyCKUwotiw",
    "63xdwScd1Ai1GigAwQxE8y",
    "6b6IMqP565TbtFFZg9iFf3",
    "65jrjEhWfAvysKfnojk1i0",
    "6Pkj4nv5K53i64cLVgkVyY",
    "2YuIyYri67bgUXKQW5V9XW",
    "5vMTfupz6z7wlNiz0xVb4j"
]

compareId = "6H8WMHCov3QGaPLbpOMpcJ"

def calculatePercentageSimilarity(song1, song2):
    similarity = {}
    total = 0
    for feature in ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness', 'time_signature', 'valence']:
        difference = abs(getattr(song1.features, feature) - getattr(song2.features, feature))
        percentage_similarity = 100 * (1 - difference)
        similarity[feature] = percentage_similarity
        total += percentage_similarity

    loudness1 = (getattr(song1.features, 'loudness') + 60) / 60
    loudness2 = (getattr(song2.features, 'loudness') + 60) / 60
    loudness_difference = abs(loudness1 - loudness2)
    loudness_similarity = 100 * (1 - loudness_difference)
    similarity['loudness'] = loudness_similarity
    total += loudness_similarity

    tempo1 = (getattr(song1.features, 'tempo') - 30) / (250 - 30)
    tempo2 = (getattr(song2.features, 'tempo') - 30) / (250 - 30)
    tempo_difference = abs(tempo1 - tempo2)
    tempo_similarity = 100 * (1 - tempo_difference)
    similarity['tempo'] = tempo_similarity
    total += tempo_similarity

    average_similarity = total / len(similarity) 
    similarity["average"] = average_similarity
    similarity["song_name"] = song1.song.song_name
    similarity["song_artist"] = song1.song.song_artist

    return similarity

access_token = authentication_service.get_auth_token(client_id=os.getenv('CLIENT_ID'), client_secret=os.getenv('CLIENT_SECRET'))

if access_token:
    headers = {
        "Authorization": f"Bearer {access_token}"
	}
    

    tracks = song_service.get_songs(gptRecommendations, headers)
    
    songs = []

    ## Get the song details of recommended songs from spotify
    for track in tracks["tracks"]:
        song = Song(track["id"], track["name"], track["artists"][0]["name"])

        songs.append(song)

    songsWithFeatures = []

    for song in songs:
        feature = song_service.get_song_features(song.song_id, headers)
        songWithFeatures = SongWithRecommendations(song, feature)

        songsWithFeatures.append(songWithFeatures)

    ## Get song we are comparing to 

    track = song_service.get_song(compareId, headers)
    compareSong = Song(track["id"], track["name"], track["artists"][0]["name"])
    compareSongFeatures = song_service.get_song_features(compareSong.song_id, headers)
    songToCompare = SongWithRecommendations(compareSong, compareSongFeatures)

    similarities = []

    for song in songsWithFeatures:
        similarity = calculatePercentageSimilarity(song, songToCompare)
        similarities.append(similarity)

    with open('similarities.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # write compared_to_song_name and compared_to_song_artist to the file
        writer.writerow(['Compared to song:', f"{songToCompare.song.song_name} by {songToCompare.song.song_artist}"])
        writer.writerow([])

        # now write the similarities table
        fieldnames = ['song_name', 'song_artist', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'time_signature', 'valence', 'average']
        table_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        table_writer.writeheader()
        for similarity in similarities:
            table_writer.writerow(similarity)

    spotWithFeatures = []
    spotRecSongs = song_service.get_song_recommendations(compareId, compareSongFeatures, headers)
    for song in spotRecSongs:
        spotFeatures = song_service.get_song_features(song.song_id, headers)
        
        spotWithFeatures.append(SongWithRecommendations(song, spotFeatures))

    spotSimilarities = []

    for song in spotWithFeatures:
        similarity = calculatePercentageSimilarity(song, songToCompare)
        spotSimilarities.append(similarity)

    with open('spotSimilarities.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # write compared_to_song_name and compared_to_song_artist to the file
        writer.writerow(['Compared to song:', f"{songToCompare.song.song_name} by {songToCompare.song.song_artist}"])
        writer.writerow([])

        # now write the similarities table
        fieldnames = ['song_name', 'song_artist', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'time_signature', 'valence', 'average']
        table_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        table_writer.writeheader()
        for similarity in spotSimilarities:
            table_writer.writerow(similarity)
else:
    print("Failed to authenticate.\n")
    exit()
