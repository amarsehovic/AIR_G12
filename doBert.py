import pickle
from transformers import DistilBertTokenizer, DistilBertModel
import torch

def load_or_generate_movie_embeddings(movies, tokenizer, model):
    try:
        with open("movie_embeddings.pkl", "rb") as f:
            embeddings = pickle.load(f)
        print("Loaded precomputed embeddings.")
    except FileNotFoundError:
        embeddings = generate_movie_embeddings(movies, tokenizer, model)
        with open("movie_embeddings.pkl", "wb") as f:
            pickle.dump(embeddings, f)
        print("Generated and saved embeddings.")
    return embeddings

def generate_movie_embeddings(movies, tokenizer, model):
    movie_embeddings = {}
    for _, row in movies.iterrows():
        movie_title = row['title']
        inputs = tokenizer(movie_title, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        movie_embeddings[row['movieId']] = embedding
    return movie_embeddings
