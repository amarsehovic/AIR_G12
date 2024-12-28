import tkinter as tk
from tkinter import messagebox
from preprocessing import preprocess_data
from transformer_model import train_transformer_model, tokenize_data
import torch


class MovieRecommendationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Personalized Movie Recommendation System")

        # Preprocess and load data
        print("Preprocessing data...")
        self.movies, self.ratings, self.user_profiles = preprocess_data()

        # Train transformer-based model
        print("Training transformer model...")
        self.transformer_model, self.tokenizer = train_transformer_model(self.ratings, self.movies)

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

    def get_user_recommendations(self, user_id, selected_genre):
        # Filter movies by genre if applicable
        filtered_movies = self.movies
        if selected_genre != "All Genres":
            filtered_movies = filtered_movies[filtered_movies['genres'].str.contains(selected_genre, na=False)]

        # Tokenize filtered movie titles
        input_ids, attention_mask = tokenize_data(filtered_movies, self.tokenizer)

        # Predict ratings for each movie
        self.transformer_model.eval()
        with torch.no_grad():
            predictions = self.transformer_model(input_ids, attention_mask).squeeze().tolist()

        # Combine predictions with movie titles
        filtered_movies['predicted_rating'] = predictions
        top_movies = filtered_movies.sort_values(by='predicted_rating', ascending=False).head(5)
        return top_movies['title'].tolist()


# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = MovieRecommendationApp(root)
    root.mainloop()
