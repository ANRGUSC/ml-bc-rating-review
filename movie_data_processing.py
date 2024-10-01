import pathlib
import pandas as pd
thisdir = pathlib.Path(__file__).parent.absolute()
# Load the movies data
movies_path = thisdir / 'data/movielens/movies.csv'
movies_df = pd.read_csv(movies_path)

# Load the ratings data
ratings_path = thisdir / 'data/movielens/ratings.csv'
ratings_df = pd.read_csv(ratings_path)

# Split the genres into a list
movies_df['genres'] = movies_df['genres'].str.split('|')

# Use the explode function to have separate rows for each genre associated with a movie
movies_exploded_df = movies_df.explode('genres')


# Merge the expanded movies dataframe with the ratings dataframe
merged_df = pd.merge(ratings_df, movies_exploded_df, on='movieId')

# Group by userId and genres and calculate the average rating
user_genre_rating_mean = merged_df.groupby(['userId', 'genres']).rating.mean().reset_index()
user_genre_pivot_mean = user_genre_rating_mean.pivot(index='userId', columns='genres', values='rating')
user_genre_pivot_mean = user_genre_pivot_mean.fillna(0)
user_genre_pivot_mean.to_csv(thisdir / 'data/movielens/user_genre_pivot_mean.csv')

user_genre_pivot_std = merged_df.groupby(['userId', 'genres']).rating.std().reset_index()
user_genre_pivot_std = user_genre_pivot_std.pivot(index='userId', columns='genres', values='rating')
user_genre_pivot_std = user_genre_pivot_std.fillna(0)
user_genre_pivot_std.to_csv(thisdir / 'data/movielens/user_genre_pivot_std.csv')



