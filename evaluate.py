import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy

# Funkcija za učitavanje embeddings
def load_embeddings(file_path='movie_embeddings.pkl'):
    """Load precomputed movie embeddings."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def create_dataset(user_profiles, movie_embeddings, ratings):
    """Create dataset using precomputed embeddings and user ratings."""
    data = []
    for _, row in user_profiles.iterrows():
        for movie_id in row['movie_list']:
            if movie_id in movie_embeddings:  # Proveri da li postoji embedding
                # Pronađi ocenu korisnika za film
                user_rating = ratings[(ratings['userId'] == row['userId']) & (ratings['movieId'] == movie_id)]
                if not user_rating.empty:
                    rating = user_rating.iloc[0]['rating']  # Prva ocena (jedna vrednost jer je dataset filtriran)
                    label = 1 if rating >= 4 else 0  # 1 za dobre ocene, 0 za loše
                else:
                    label = 0  # Ako nema ocene, možeš postaviti podrazumevanu vrednost (npr. 0)

                # Dodaj u dataset
                data.append({
                    "embedding": movie_embeddings[movie_id],
                    "label": label
                })
    return pd.DataFrame(data)  # Vraća pandas DataFrame


# Učitaj embeddings
print("Loading precomputed movie embeddings...")
movie_embeddings = load_embeddings()

# Load preprocessed data
from preprocessing import preprocess_data
movies, ratings, user_profiles = preprocess_data(testing=False)

# Podela korisničkih profila na trening i evaluacioni skup
train_user_profiles = user_profiles.iloc[:int(0.8 * len(user_profiles))]
eval_user_profiles = user_profiles.iloc[int(0.8 * len(user_profiles)):]

# Kreiraj evaluacioni skup sa stvarnim label-ima
eval_dataset = create_dataset(eval_user_profiles, movie_embeddings, ratings)
train_dataset = create_dataset(train_user_profiles, movie_embeddings, ratings)

# Pripremi podatke za trening i evaluaciju
X_train = list(train_dataset["embedding"])  # Embeddings za trening
y_train = list(train_dataset["label"])  # Label za trening

X_eval = list(eval_dataset["embedding"])  # Embeddings za evaluaciju
y_true = list(eval_dataset["label"])  # Label za evaluaciju

# --------- Random Forest ---------
print("Training Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Generiši predikcije za Random Forest
rf_y_pred = rf_model.predict(X_eval)

# Evaluacija Random Forest modela
rf_precision = precision_score(y_true, rf_y_pred, average='weighted', zero_division=1)
rf_recall = recall_score(y_true, rf_y_pred, average='weighted', zero_division=1)
rf_f1 = f1_score(y_true, rf_y_pred, average='weighted', zero_division=1)

print("\nRandom Forest Evaluation Metrics:")
print(f"Precision: {rf_precision:.4f}")
print(f"Recall: {rf_recall:.4f}")
print(f"F1 Score: {rf_f1:.4f}")

print("\nRandom Forest Classification Report:")
print(classification_report(y_true, rf_y_pred, zero_division=1))


# --------- Collaborative Filtering ---------
print("\nTraining Collaborative Filtering model...")
reader = Reader(rating_scale=(0.5, 5.0))  # Skaliranje ocena
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Podela na trening i test skup
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Treniraj Item-based CF model
sim_options = {'name': 'cosine', 'user_based': False}  # Item-based
cf_model = KNNBasic(sim_options=sim_options)
cf_model.fit(trainset)

# Generiši predikcije za test skup
cf_predictions = cf_model.test(testset)

# Evaluacija CF modela
print("\nCollaborative Filtering RMSE:", accuracy.rmse(cf_predictions))
print("Collaborative Filtering MAE:", accuracy.mae(cf_predictions))

# Pretvori predikcije u binarne oznake
cf_y_pred = [1 if pred.est >= 4 else 0 for pred in cf_predictions]
cf_y_true = [1 if true_r >= 4 else 0 for (_, _, true_r, _, _) in cf_predictions]

# Evaluacija sa binarnim oznakama
cf_precision = precision_score(cf_y_true, cf_y_pred, average='weighted', zero_division=1)
cf_recall = recall_score(cf_y_true, cf_y_pred, average='weighted', zero_division=1)
cf_f1 = f1_score(cf_y_true, cf_y_pred, average='weighted', zero_division=1)

print("\nCollaborative Filtering Evaluation Metrics:")
print(f"Precision: {cf_precision:.4f}")
print(f"Recall: {cf_recall:.4f}")
print(f"F1 Score: {cf_f1:.4f}")

print("\nCollaborative Filtering Classification Report:")
print(classification_report(cf_y_true, cf_y_pred, zero_division=1))
