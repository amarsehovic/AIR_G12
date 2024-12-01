import tkinter as tk
from tkinter import messagebox
from preprocessing import preprocess_data  # Import preprocessing function
from model_training import train_model     # Import model training function


class MovieRecommendationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Personalized Movie Recommendation System")

        # Preprocess and load data
        print("Preprocessing data...")
        self.movies, self.ratings, self.train, self.test, self.user_profiles = preprocess_data()

        # Train the recommendation model
        print("Training model...")
        self.model = train_model(self.ratings)

        # User ID Input
        self.label_user_id = tk.Label(root, text="Enter User ID:")
        self.label_user_id.pack(pady=5)

        self.entry_user_id = tk.Entry(root)
        self.entry_user_id.pack(pady=5)

        # Fetch Recommendations Button
        self.fetch_button = tk.Button(root, text="Get Recommendations", command=self.fetch_recommendations)
        self.fetch_button.pack(pady=10)

        # Recommendations Listbox
        self.label_recommendations = tk.Label(root, text="Top-5 Recommendations:")
        self.label_recommendations.pack(pady=5)

        self.listbox_recommendations = tk.Listbox(root, height=5, width=50)
        self.listbox_recommendations.pack(pady=5)

    def fetch_recommendations(self):
        # Get input user ID
        user_id = self.entry_user_id.get()
        if not user_id.isdigit():
            messagebox.showerror("Error", "Please enter a valid numeric User ID.")
            return

        user_id = int(user_id)

        # Fetch personalized recommendations
        recommendations = self.get_user_recommendations(user_id)

        # Clear the listbox and add new recommendations
        self.listbox_recommendations.delete(0, tk.END)
        if not recommendations:
            self.listbox_recommendations.insert(tk.END, "No recommendations available.")
        else:
            for movie in recommendations:
                self.listbox_recommendations.insert(tk.END, movie)

    def get_user_recommendations(self, user_id):
        # Check if the user exists in the preprocessed user profiles
        if user_id not in self.user_profiles['userId'].values:
            # Fallback: Recommend top-rated movies for new users
            top_movies = self.ratings.groupby('movieId')['rating'].mean().sort_values(ascending=False).head(5)
            return self.movies[self.movies['movieId'].isin(top_movies.index)]['title'].tolist()

        # Fetch the user's profile
        user_profile = self.user_profiles[self.user_profiles['userId'] == user_id]
        liked_movies = user_profile['movie_list'].iloc[0]

        # Predict ratings for all movies the user hasn't rated
        all_movies = self.movies['movieId'].unique()
        user_rated_movies = self.ratings[self.ratings['userId'] == user_id]['movieId'].unique()
        unrated_movies = [m for m in all_movies if m not in user_rated_movies]

        # Predict ratings for unrated movies
        predictions = [
            (movie, self.model.predict(user_id, movie).est) for movie in unrated_movies
        ]

        # Sort by predicted rating and recommend top 5
        recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)[:5]
        return [self.movies[self.movies['movieId'] == r[0]]['title'].iloc[0] for r in recommendations]


# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = MovieRecommendationApp(root)
    root.mainloop()
