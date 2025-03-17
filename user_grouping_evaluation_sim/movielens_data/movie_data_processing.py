import pathlib
from typing import Tuple

import pandas as pd


def load_data(base_dir: pathlib.Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load movies and ratings data from CSV files.
    
    Args:
        base_dir: Base directory containing the data files
        
    Returns:
        Tuple containing (movies_dataframe, ratings_dataframe)
    """
    movies_path = base_dir / 'movies.csv'
    ratings_path = base_dir / 'ratings.csv'
    
    movies_df = pd.read_csv(movies_path)
    ratings_df = pd.read_csv(ratings_path)
    
    return movies_df, ratings_df

def create_user_genre_matrices(movies_df: pd.DataFrame, ratings_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create user-genre rating matrices with mean and standard deviation calculations.
    
    Args:
        movies_df: The movies dataframe
        ratings_df: The ratings dataframe
        
    Returns:
        Tuple containing (mean_ratings_pivot, std_ratings_pivot)
    """
    # Preprocess movies data
    movies_df['genres'] = movies_df['genres'].str.split('|')

    # Use the explode function to have separate rows for each genre associated with a movie
    movies_exploded_df = movies_df.explode('genres')
    
    # Merge the expanded movies dataframe with the ratings dataframe
    merged_df = pd.merge(ratings_df, movies_exploded_df, on='movieId')
    
    # Calculate mean ratings per user-genre
    user_genre_rating_mean = merged_df.groupby(['userId', 'genres'])['rating'].mean().reset_index()
    user_genre_pivot_mean = user_genre_rating_mean.pivot(
        index='userId', 
        columns='genres', 
        values='rating'
    ).fillna(0)
    
    # Calculate standard deviation of ratings per user-genre
    user_genre_rating_std = merged_df.groupby(['userId', 'genres'])['rating'].std().reset_index()
    user_genre_pivot_std = user_genre_rating_std.pivot(
        index='userId', 
        columns='genres', 
        values='rating'
    ).fillna(0)
    
    return user_genre_pivot_mean, user_genre_pivot_std


def save_matrices(mean_matrix: pd.DataFrame, std_matrix: pd.DataFrame, output_dir: pathlib.Path) -> None:
    """
    Save the user-genre matrices to CSV files.
    
    Args:
        mean_matrix: User-genre mean ratings matrix
        std_matrix: User-genre rating standard deviations matrix
        output_dir: Directory to save the output files
    """
    mean_matrix.to_csv(output_dir / 'user_genre_pivot_mean.csv')
    std_matrix.to_csv(output_dir / 'user_genre_pivot_std.csv')
    print(f"Files saved to: {output_dir}")


def main() -> None:
    """Main function to orchestrate the data processing pipeline."""
    # Get the directory containing this script
    thisdir = pathlib.Path(__file__).parent.absolute()
    
    # Load data
    movies_df, ratings_df = load_data(thisdir)
    
    # Create user-genre matrices
    mean_matrix, std_matrix = create_user_genre_matrices(movies_df, ratings_df)
    
    # Save results
    save_matrices(mean_matrix, std_matrix, thisdir)


if __name__ == "__main__":
    main()

