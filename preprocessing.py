import pandas as pd
import numpy as np


def preprocess_data(movies_path='ml-latest-small/movies.csv', ratings_path='ml-latest-small/ratings.csv'):
    try:
        print("Loading datasets...")
        movies = pd.read_csv(movies_path)
        ratings = pd.read_csv(ratings_path)
    except FileNotFoundError as e:
        print(f"Error: One or more files are missing. Please check the paths: {e}")
        return None, None, None
    except pd.errors.EmptyDataError:
        print("Error: One or more files are empty.")
        return None, None, None
    except Exception as e:
        print(f"Unexpected error while loading datasets: {e}")
        return None, None, None

    try:
        print("Cleaning data...")
        # find if any data is missing
        if movies[['movieId', 'title']].isnull().any().any():
            print("Issue found: Missing data in movies (movieId or title). Dropping rows.")
            movies.dropna(subset=['movieId', 'title'], inplace=True)

        if ratings[['userId', 'movieId', 'rating']].isnull().any().any():
            print("Issue found: Missing ratings data. Dropping rows.")
            ratings.dropna(subset=['userId', 'movieId', 'rating'], inplace=True)

        # make sure that ratings are in correct range: 0.5 to 5
        ratings = ratings[ratings['rating'].between(0.5, 5.0)]

    except KeyError as e:
        print(f"Error: Missing expected columns in the dataset: {e}")
        return None, None, None
    except Exception as e:
        print(f"Unexpected error during data cleaning: {e}")
        return None, None, None

    # removing users and movies with not enough interaction
    try:
        min_user_ratings = 5
        min_movie_ratings = 10
        user_counts = ratings['userId'].value_counts()
        movie_counts = ratings['movieId'].value_counts()
        ratings = ratings[ratings['userId'].isin(user_counts[user_counts >= min_user_ratings].index)]
        ratings = ratings[ratings['movieId'].isin(movie_counts[movie_counts >= min_movie_ratings].index)]
    except Exception as e:
        print(f"Error during filtering users and movies: {e}")
        return None, None, None

    #transform timpestamps
    try:
        print("Transforming data...")
        ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
        ratings['year'] = ratings['timestamp'].dt.year
        ratings['month'] = ratings['timestamp'].dt.month
    except Exception as e:
        print(f"Error during data transformation: {e}")
        return None, None, None

    try:
        print("Creating user profiles...")
        user_profiles = ratings.groupby('userId').agg({
            'rating': ['mean', 'count'],
            'movieId': lambda x: list(x)
        })
        user_profiles.columns = ['avg_rating', 'rating_count', 'movie_list']
        user_profiles.reset_index(inplace=True)
    except Exception as e:
        print(f"Error while creating user profiles: {e}")
        return None, None, None

    print("Preprocessing complete.")
    return movies, ratings, user_profiles


def check_data_integrity(movies, ratings):
    if not set(ratings['movieId']).issubset(set(movies['movieId'])):
        print("Warning: Some movie IDs in ratings don't exist in the movies dataset.")

    if ratings['userId'].min() < 0:
        print("Warning: Found negative userId in ratings.")

    if ratings['movieId'].min() < 0:
        print("Warning: Found negative movieId in ratings.")


def preprocess():
    movies, ratings, user_profiles = preprocess_data()
    if movies is None or ratings is None or user_profiles is None:
        return None, None, None
    check_data_integrity(movies, ratings)

    return movies, ratings, user_profiles
