from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split


class TransformerRecommender(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=1):
        super(TransformerRecommender, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, movie_embeddings):
        hidden = self.relu(self.fc1(movie_embeddings))
        ratings = self.out(hidden)
        return ratings


# Tokenize input data
def tokenize_data(movies, tokenizer, max_length=64):
    """
    Tokenizes movie titles using the provided tokenizer.

    Args:
        movies (DataFrame): A DataFrame containing movie titles.
        tokenizer (BertTokenizer): A tokenizer for converting text to token IDs.
        max_length (int): Maximum length for tokenized sequences.

    Returns:
        tuple: Token IDs and attention masks as PyTorch tensors.
    """
    tokens = tokenizer(
        movies['title'].tolist(),
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    return tokens['input_ids'], tokens['attention_mask']


# Precompute movie embeddings using BERT
def precompute_movie_embeddings(movies, tokenizer, bert_model, batch_size=16):
    print("Precomputing embeddings for movie titles in batches...")
    movie_titles = movies['title'].tolist()
    all_embeddings = []

    for i in range(0, len(movie_titles), batch_size):
        batch_titles = movie_titles[i:i + batch_size]
        tokens = tokenizer(
            batch_titles,
            max_length=64,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        with torch.no_grad():
            outputs = bert_model(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'])
            embeddings = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token embeddings
            all_embeddings.append(embeddings)

    # Concatenate all batch embeddings
    return torch.cat(all_embeddings, dim=0)


# Train the Transformer model with precomputed embeddings
def train_transformer_model(ratings, movies, epochs=3, batch_size=16, lr=1e-4):
    print("Initializing tokenizer and pretrained BERT model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    # Precompute embeddings
    movie_embeddings = precompute_movie_embeddings(movies, tokenizer, bert_model)

    # Use a subset of data
    train_ratings, val_ratings = train_test_split(ratings.sample(frac=0.1, random_state=42), test_size=0.2)

    print("Initializing recommendation model...")
    model = TransformerRecommender(input_size=movie_embeddings.size(1))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print("Starting training...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, len(train_ratings), batch_size):
            batch = train_ratings.iloc[i:i + batch_size]
            movie_map = {mid: idx for idx, mid in enumerate(movies['movieId'])}
            movie_idxs = batch['movieId'].map(movie_map).tolist()
            ratings = torch.tensor(batch['rating'].values, dtype=torch.float32).unsqueeze(1)

            # Use precomputed embeddings
            batch_embeddings = movie_embeddings[movie_idxs]

            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_embeddings)
            loss = criterion(outputs, ratings)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

    return model, tokenizer