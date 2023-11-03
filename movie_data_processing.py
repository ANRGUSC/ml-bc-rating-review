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
user_genre_rating = merged_df.groupby(['userId', 'genres']).rating.mean().reset_index()

# Pivot the table to have genres as columns and users as rows with their average rating for that genre
user_genre_pivot = user_genre_rating.pivot(index='userId', columns='genres', values='rating')

# Fill the missing values with 0
user_genre_pivot = user_genre_pivot.fillna(0)

print(user_genre_pivot.head())

# Save the user_genre_pivot dataframe to a csv file
user_genre_pivot.to_csv(thisdir / 'data/movielens/user_genre_pivot.csv')




