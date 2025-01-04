import tkinter as tk
from tkinter import messagebox
from preprocessing import preprocess_data
from model_training import train_model, evaluate_model
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommendationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Personalized Movie Recommendation System")

        # Preprocess and load data
        print("Preprocessing data...")
        self.movies, self.ratings, self.user_profiles = preprocess_data()

        # Load DistilBERT model and tokenizer
        print("Loading DistilBERT model...")
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")

        # Precompute movie embeddings (distilBERT for movie descriptions)
        print("Generating movie embeddings...")
        self.movie_embeddings = self.generate_movie_embeddings()

        # Train the recommendation model
        print("Training model...")
        self.model_recommendation, self.testset = train_model(self.ratings)
        evaluate_model(self.model_recommendation, self.testset, k=5)

        # User ID Input
        self.label_user_id = tk.Label(root, text="Enter User ID:")
        self.label_user_id.pack(pady=5)

        self.entry_user_id = tk.Entry(root)
        self.entry_user_id.pack(pady=5)

        # Genre Dropdown
        self.label_genre = tk.Label(root, text="Filter by Genre:")
        self.label_genre.pack(pady=5)

        self.genre_var = tk.StringVar(value="All Genres")
        genres = ["All Genres"] + sorted(self.movies['genres'].str.split('|').explode().dropna().unique())
        self.genre_dropdown = tk.OptionMenu(root, self.genre_var, *genres)
        self.genre_dropdown.pack(pady=5)

        # Fetch Recommendations Button
        self.fetch_button = tk.Button(root, text="Get Recommendations", command=self.fetch_recommendations)
        self.fetch_button.pack(pady=10)

        # Recommendations Listbox
        self.label_recommendations = tk.Label(root, text="Top-5 Recommendations:")
        self.label_recommendations.pack(pady=5)

        self.listbox_recommendations = tk.Listbox(root, height=5, width=50)
        self.listbox_recommendations.pack(pady=5)

    def fetch_recommendations(self):
        user_id = self.entry_user_id.get()
        selected_genre = self.genre_var.get()

        if not user_id.isdigit():
            messagebox.showerror("Error", "Please enter a valid numeric User ID.")
            return

        user_id = int(user_id)

        # Fetch personalized recommendations
        recommendations = self.get_user_recommendations(user_id, selected_genre)

        # Clear the listbox and add new recommendations
        self.listbox_recommendations.delete(0, tk.END)
        if not recommendations:
            self.listbox_recommendations.insert(tk.END, "No recommendations available.")
        else:
            for movie in recommendations:
                self.listbox_recommendations.insert(tk.END, movie)

    def generate_movie_embeddings(self):
        embeddings = {}
        for index, row in self.movies.iterrows():
            description = row['title']  # Assuming movies have a 'description' field
            embedding = self.get_movie_embedding(description)
            embeddings[row['movieId']] = embedding
        return embeddings

    def get_movie_embedding(self, description):
        # Tokenize the movie description
        inputs = self.tokenizer(description, return_tensors="pt", truncation=True, padding=True, max_length=512)

        # Get the embeddings from DistilBERT
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Return the embedding of the [CLS] token (first token of the sequence)
        embedding = outputs.last_hidden_state[:, 0, :]
        return embedding.numpy().flatten()

    def get_user_recommendations(self, user_id, selected_genre):
        # Filter movies by genre if a genre is selected
        filtered_movies = self.movies
        if selected_genre != "All Genres":
            filtered_movies = filtered_movies[filtered_movies['genres'].str.contains(selected_genre, na=False)]

        # Check if the user exists in the preprocessed user profiles
        if user_id not in self.user_profiles['userId'].values:
            # Fallback: Recommend top-rated movies for new users
            top_movies = self.ratings.groupby('movieId')['rating'].mean().sort_values(ascending=False).head(5)
            top_movies = filtered_movies[filtered_movies['movieId'].isin(top_movies.index)]
            return top_movies['title'].tolist()

        # Fetch the user's profile
        user_profile = self.user_profiles[self.user_profiles['userId'] == user_id]
        liked_movies = user_profile['movie_list'].iloc[0]

        # Predict ratings for unrated movies
        unrated_movies = filtered_movies[~filtered_movies['movieId'].isin(liked_movies)]['movieId']
        predictions = [
            (movie, self.model_recommendation.predict(user_id, movie).est) for movie in unrated_movies
        ]

        # Sort movies by predicted rating and recommend the top 5
        recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)[:5]

        # Enhance recommendations by considering movie descriptions
        recommendations_with_content = self.enhance_with_content_based_recommendations(recommendations)
        return [movie[0] for movie in recommendations_with_content]

    def enhance_with_content_based_recommendations(self, recommendations):
        # Get the embeddings of the recommended movies
        recommended_movie_ids = [movie[0] for movie in recommendations]
        recommended_movie_embeddings = [self.movie_embeddings[movie_id] for movie_id in recommended_movie_ids]

        # Calculate similarity between recommended movies and other available movies
        similarity_scores = []
        for movie_id, embedding in self.movie_embeddings.items():
            similarity = self.compute_similarity(embedding, recommended_movie_embeddings)
            similarity_scores.append((movie_id, similarity))

        # Sort movies based on similarity and return top 5
        sorted_similarities = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[:5]
        return [(self.movies[self.movies['movieId'] == movie_id]['title'].iloc[0], score) for movie_id, score in
                sorted_similarities]

    def compute_similarity(self, embedding, recommended_embeddings):
        # Compute cosine similarity between embeddings
        similarities = []
        for rec_embedding in recommended_embeddings:
            similarity = cosine_similarity([embedding], [rec_embedding])
            similarities.append(similarity[0][0])
        return np.mean(similarities)


# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = MovieRecommendationApp(root)
    root.mainloop()
