import pandas as pd

def preprocess_data(testing=False):
    # Load datasets
    print("Loading datasets...")
    movies = pd.read_csv('ml-latest-small/movies.csv')
    small_ratings = pd.read_csv('ml-latest-small/ratings.csv')

    # Data Cleaning
    print("Cleaning data...")
    small_ratings.dropna(inplace=True)
    movies.dropna(inplace=True)
    small_ratings = small_ratings[small_ratings['rating'].between(0.5, 5.0)]

    # Filter out users and movies with very few interactions
    min_user_ratings = 5
    min_movie_ratings = 10
    user_counts = small_ratings['userId'].value_counts()
    movie_counts = small_ratings['movieId'].value_counts()
    small_ratings = small_ratings[small_ratings['userId'].isin(user_counts[user_counts >= min_user_ratings].index)]
    small_ratings = small_ratings[small_ratings['movieId'].isin(movie_counts[movie_counts >= min_movie_ratings].index)]

    # Data Transformation
    print("Transforming data...")
    small_ratings['timestamp'] = pd.to_datetime(small_ratings['timestamp'], unit='s')

    # Feature Engineering: User Profiles
    print("Creating user profiles...")
    user_profiles = small_ratings.groupby('userId').agg({
        'rating': ['mean', 'count'],
        'movieId': lambda x: list(x)
    })
    user_profiles.columns = ['avg_rating', 'rating_count', 'movie_list']
    user_profiles.reset_index(inplace=True)

    # Reduce dataset for testing if the testing flag is True
    if testing:
        print("Reducing dataset for testing...")
        small_ratings = small_ratings.sample(frac=0.01, random_state=42)
        movies = movies.head(100)

    print("Preprocessing complete.")
    return movies, small_ratings, user_profiles


if __name__ == "__main__":
    movies, ratings, user_profiles = preprocess_data(testing=True)  # Set testing to True for reduced dataset
    print(movies.head(), ratings.head(), user_profiles.head())
