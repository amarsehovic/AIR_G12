import tkinter as tk
from tkinter import messagebox, ttk
from transformers import DistilBertTokenizer, DistilBertModel
from doBert import load_or_generate_movie_embeddings, generate_movie_embeddings
from preprocessing import preprocess_data
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image, ImageTk


class MovieRecommendationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Movie Recommendation GUI")

        # Header Frame
        self.header_frame = tk.Frame(root, bg="white", height=80)
        self.header_frame.pack(fill="x", pady=5)

        # Add Logo
        logo_path = "cinema2.png"  # Putanja do logotipa
        logo_image = Image.open(logo_path)  # Učitajte sliku
        resized_logo = logo_image.resize((50, 50), Image.LANCZOS)  # Koristimo LANCZOS za visok kvalitet skaliranja
        self.logo_image = ImageTk.PhotoImage(resized_logo)  # Konvertujte za tkinter
        self.logo_label = tk.Label(self.header_frame, image=self.logo_image, bg="white")
        self.logo_label.pack(side="left", padx=10)

        # Add App Title
        self.title_label = tk.Label(self.header_frame, text="Personalized Movie Recommendation System", bg="white", fg="black",
                                    font=("Arial", 16, "bold"))
        self.title_label.pack(side="left", padx=10)

        # Main Content Frame (split into left and right)
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Left Frame
        self.left_frame = tk.Frame(self.main_frame, width=400)
        self.left_frame.pack(side="left", fill="y", expand=True, padx=10)

        # Right Frame
        self.right_frame = tk.Frame(self.main_frame, width=400, bg="#f7f7f7")
        self.right_frame.pack(side="right", fill="both", expand=True, padx=10)

        # Preprocess and load data
        print("Preprocessing data...")
        self.movies, self.ratings, self.user_profiles = preprocess_data()

        # Load DistilBERT model and tokenizer
        print("Loading DistilBERT model...")
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")

        # Precompute or load precomputed movie embeddings
        print("Loading or generating movie embeddings...")
        #self.movie_embeddings = generate_movie_embeddings(self.movies, self.tokenizer, self.model)
        self.movie_embeddings = load_or_generate_movie_embeddings(self.movies, self.tokenizer, self.model)

        # ----- Left Frame Content -----
        label_width = 20
        dropdown_width = 18
        label_color = "#222222"

        # select User ID
        self.label_user_id = tk.Label(self.left_frame, text="Select User ID:", anchor="w", width=label_width, fg=label_color)
        self.label_user_id.pack(pady=(15, 5), padx=5, anchor="w")

        self.user_id_var = tk.StringVar()
        user_ids = sorted(self.user_profiles['userId'].astype(str).tolist())  # Konvertujemo u string za dropdown
        self.user_id_dropdown = ttk.Combobox(self.left_frame, textvariable=self.user_id_var, values=user_ids, state="readonly", width=dropdown_width)
        self.user_id_dropdown.pack(pady=5, padx=5, anchor="w")

        # Genre Dropdown
        self.label_genre = tk.Label(self.left_frame, text="Filter movies by Genre:", anchor="w", width=label_width, fg=label_color)
        self.label_genre.pack(pady=(25, 5), padx=5, anchor="w")

        self.genre_var = tk.StringVar(value="All Genres")
        genres = ["All Genres"] + sorted(self.movies['genres'].str.split('|').explode().dropna().unique())
        self.genre_dropdown = tk.OptionMenu(self.left_frame, self.genre_var, *genres)
        self.genre_dropdown.config(width=dropdown_width)  # Uniformna širina za dropdown meni
        self.genre_dropdown.pack(pady=5, padx=5, anchor="w")

        # Number of Recommendations (Checkboxes)
        self.label_top_n = tk.Label(self.left_frame, text="How many recommendations do you want to get?", fg=label_color)
        self.label_top_n.pack(pady=(25, 5), padx=5, anchor="w")

        self.top_n_var = tk.IntVar(value=5)  # Default value: Top 5

        self.checkbox_top_3 = tk.Checkbutton(self.left_frame, text="Top 3", variable=self.top_n_var, onvalue=3, offvalue=0, fg=label_color)
        self.checkbox_top_3.pack(pady=2, padx=0, anchor="w")

        self.checkbox_top_5 = tk.Checkbutton(self.left_frame, text="Top 5", variable=self.top_n_var, onvalue=5, offvalue=0, fg=label_color)
        self.checkbox_top_5.pack(pady=2, padx=0, anchor="w")

        self.checkbox_top_10 = tk.Checkbutton(self.left_frame, text="Top 10", variable=self.top_n_var, onvalue=10, offvalue=0, fg=label_color)
        self.checkbox_top_10.pack(pady=2, padx=0, anchor="w")

        # Fetch Recommendations Button
        self.fetch_button = tk.Button(self.left_frame, text="Recommend Movies", command=self.fetch_recommendations, bg="#28a745", fg="white", font=("Arial", 12, "bold"))
        self.fetch_button.pack(pady=10, side="bottom")

        # ----- Right Frame Content -----
        # Recommendations Listbox
        self.label_recommendations = tk.Label(self.right_frame, text="Recommendations", font=("Arial", 12, "bold"), bg="#ff4155")
        self.label_recommendations.pack(pady=5)

        self.listbox_recommendations = tk.Listbox(self.right_frame, height=20, width=60)
        self.listbox_recommendations.pack(pady=5)

        # ------ Footer -------
        # Footer Frame
        self.footer_frame = tk.Frame(root, bg="#f0f0f0", height=30)  # Svetlosiva pozadina za footer
        self.footer_frame.pack(fill="x", side="bottom", pady=5)

        self.footer_label = tk.Label(self.footer_frame, text="© Copyright", bg="#f0f0f0", fg="#333333",
                                     font=("Helvetica", 10))
        self.footer_label.pack(pady=5)

    def fetch_recommendations(self):
        user_id = self.user_id_var.get()
        genre = self.genre_var.get()
        top_n = self.top_n_var.get()

        # Update the recommendations label based on the number of movies selected
        if top_n == 3:
            self.label_recommendations.config(text="Your top 3 movies to watch next")
        elif top_n == 5:
            self.label_recommendations.config(text="Your top 5 movies to watch next")
        elif top_n == 10:
            self.label_recommendations.config(text="Your top 10 movies to watch next")

        # Get well-rated movies for the selected user
        user_ratings = self.ratings[self.ratings['userId'] == int(user_id)]
        well_rated_movies = user_ratings[user_ratings['rating'] >= 4.0]  # Filter high ratings (>= 4.0)

        # If no well-rated movies, display message
        if well_rated_movies.empty:
            messagebox.showinfo("No Data", "This user has no well-rated movies.")
            return

        # Get movie IDs of well-rated movies
        well_rated_movie_ids = well_rated_movies['movieId'].tolist()

        # Filter movies by selected genre
        if genre != "All Genres":
            genre_movies = self.movies[self.movies['genres'].str.contains(genre)]
        else:
            genre_movies = self.movies

        # Filter out movies the user has already rated
        unseen_movies = genre_movies[~genre_movies['movieId'].isin(well_rated_movie_ids)]

        # Get movie embeddings for unseen movies
        unseen_movie_embeddings = {movie_id: self.movie_embeddings[movie_id] for movie_id in unseen_movies['movieId']}

        # Get embeddings for the user's well-rated movies
        well_rated_movie_embeddings = {movie_id: self.movie_embeddings[movie_id] for movie_id in well_rated_movie_ids}

        # Calculate cosine similarity between well-rated movies and unseen movies
        recommendations = self.get_similar_movies(well_rated_movie_embeddings, unseen_movie_embeddings, top_n)

        # Display recommendations in the listbox
        self.listbox_recommendations.delete(0, tk.END)  # Clear previous recommendations
        for index, (movie_id, similarity) in enumerate(recommendations, start=1):
            movie_title = self.movies[self.movies['movieId'] == movie_id]['title'].values[0]
            self.listbox_recommendations.insert(tk.END, f"{index}. {movie_title}")

    def get_similar_movies(self, well_rated_movie_embeddings, unseen_movie_embeddings, top_n):
        movie_similarities = []

        # Compare each unseen movie with well-rated movies
        for unseen_movie_id, unseen_embedding in unseen_movie_embeddings.items():
            for well_rated_movie_id, well_rated_embedding in well_rated_movie_embeddings.items():
                similarity = cosine_similarity([unseen_embedding], [well_rated_embedding])[0][0]
                movie_similarities.append((unseen_movie_id, similarity))

        # Sort by similarity and return the top N recommendations
        movie_similarities.sort(key=lambda x: x[1], reverse=True)
        return movie_similarities[:top_n]


# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = MovieRecommendationApp(root)
    root.mainloop()
