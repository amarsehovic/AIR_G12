import tkinter as tk
from tkinter import messagebox, ttk
from preprocessing import preprocess_data
from model_training import train_model
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

        # Train the recommendation model
        print("Training model...")
        self.model, self.testset = train_model(self.ratings)

        # ----- Left Frame Content -----
        label_width = 20
        dropdown_width = 18
        label_color = "#222222"

        # select User ID
        self.label_user_id = tk.Label(self.left_frame, text="Select User ID:" , anchor="w", width=label_width, fg=label_color)
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
        selected_genre = self.genre_var.get()
        top_n = self.top_n_var.get()


        if not user_id.isdigit():
            messagebox.showerror("Error", "Please select a valid numeric User ID.")
            return

        user_id = int(user_id)

        # Update the recommendations label dynamically
        self.label_recommendations.config(text=f"Top {top_n} movies:")

        # Fetch personalized recommendations
        recommendations = self.get_user_recommendations(user_id, selected_genre, top_n)

        # Clear the listbox and add new recommendations
        self.listbox_recommendations.delete(0, tk.END)
        if not recommendations:
            self.listbox_recommendations.insert(tk.END, "No recommendations available.")
        else:
            for index, movie in enumerate(recommendations, start=1):  # Add numbering (1, 2, 3, ...)
                self.listbox_recommendations.insert(tk.END, f"{index}. {movie}")

    def get_user_recommendations(self, user_id, selected_genre, top_n):
        # Filter movies by genre if a genre is selected
        filtered_movies = self.movies
        if selected_genre != "All Genres":
            filtered_movies = filtered_movies[filtered_movies['genres'].str.contains(selected_genre, na=False)]

        # Check if the user exists in the preprocessed user profiles
        if user_id not in self.user_profiles['userId'].values:
            # Fallback: Recommend top-rated movies for new users
            top_movies = self.ratings.groupby('movieId')['rating'].mean().sort_values(ascending=False).head(top_n)
            top_movies = filtered_movies[filtered_movies['movieId'].isin(top_movies.index)]
            return top_movies['title'].tolist()

        # Fetch the user's profile
        user_profile = self.user_profiles[self.user_profiles['userId'] == user_id]
        liked_movies = user_profile['movie_list'].iloc[0]

        # Predict ratings for unrated movies
        unrated_movies = filtered_movies[~filtered_movies['movieId'].isin(liked_movies)]['movieId']
        predictions = [
            (movie, self.model.predict(user_id, movie).est) for movie in unrated_movies
        ]

        # Sort movies by predicted rating and recommend the top N
        recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]
        return [self.movies[self.movies['movieId'] == movie_id]['title'].iloc[0] for movie_id, _ in recommendations]


# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = MovieRecommendationApp(root)
    root.mainloop()
