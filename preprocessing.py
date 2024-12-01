import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_data():
    # Load the datasets
    movies = pd.read_csv('ml-32m/movies.csv')
    ratings = pd.read_csv('ml-32m/ratings.csv')

    # Data Cleaning
    print("Cleaning Data...")
    # Remove missing or invalid entries
    ratings.dropna(inplace=True)
    movies.dropna(inplace=True)

    # Filter out users and movies with very few interactions
    min_user_ratings = 5
    min_movie_ratings = 10
    user_counts = ratings['userId'].value_counts()
    movie_counts = ratings['movieId'].value_counts()
    ratings = ratings[ratings['userId'].isin(user_counts[user_counts >= min_user_ratings].index)]
    ratings = ratings[ratings['movieId'].isin(movie_counts[movie_counts >= min_movie_ratings].index)]

    # Data Transformation
    print("Transforming Data...")
    # Parse timestamps into year and month
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
    ratings['year'] = ratings['timestamp'].dt.year
    ratings['month'] = ratings['timestamp'].dt.month

    # Encode genres using one-hot encoding
    print("Encoding genres...")
    genre_dummies = movies['genres'].str.get_dummies(sep='|')
    movies = pd.concat([movies, genre_dummies], axis=1)

    # Feature Engineering: Create user profiles
    print("Creating user profiles...")
    user_profiles = ratings.groupby('userId').agg({
        'rating': ['mean', 'count'],
        'movieId': lambda x: list(x)
    })
    user_profiles.columns = ['avg_rating', 'rating_count', 'movie_list']
    user_profiles.reset_index(inplace=True)

    # Data Splitting
    print("Splitting data...")
    train, test = train_test_split(ratings, test_size=0.2, random_state=42)

    # Return processed data
    return movies, ratings, train, test, user_profiles


# Run preprocessing
if __name__ == "__main__":
    movies, ratings, train, test, user_profiles = preprocess_data()
    print("Preprocessing complete.")
    print(f"Movies: {movies.shape}")
    print(f"Ratings: {ratings.shape}")
    print(f"Train: {train.shape}, Test: {test.shape}")
    print(user_profiles.head())
