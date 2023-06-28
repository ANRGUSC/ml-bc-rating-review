import random
from typing import Dict

ATTRIBUTES = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness', 'valence', 'popularity']

ATTRIBUTES_DESCRIPTION = """
    Acousticness describes how acoustic a song is. A score of 1.0 means the song is most likely to be an acoustic one.
    Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.
    Energy represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy.
    The instrumentalness value represents the amount of vocals in the song. The closer it is to 1.0, the more instrumental the song is.
    Liveness, when above 0.8 provides strong likelihood that the track is live.
    Speechiness detects the presence of spoken words in a track. If the speechiness of a song is above 0.66, it is probably made of spoken words, a score between 0.33 and 0.66 is a song that may contain both music and words, and a score below 0.33 means the song does not have any speech.
    Valence is a measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative.
    The popularity represents the popularity of the album the song comes from. The value will be between 0 and 100, with 100 being the most popular. 
"""

def get_random_song_attributes() -> Dict[str, float]:
    """Gets a random set of song attributes"""
    attributes = {attr: round(random.random(), 3) for attr in ATTRIBUTES}
    attributes["popularity"] = random.randint(0, 100)
    return attributes

def calculatePercentageSimilarity(attribute1, attribute2):
    similarity = {}
    total = 0
    for feature in ATTRIBUTES:            
        difference = abs(attribute1[feature] - attribute2[feature])
        percentage_similarity = 100 - difference if feature == 'popularity' else 100 * (1 - difference)

        similarity[feature] = percentage_similarity
        total += percentage_similarity

    average_similarity = total / len(similarity) 
    similarity["average"] = average_similarity

    return similarity
