from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse

# Function to train the model
def train_model(ratings):
    print("Preparing data for training...")
    # Prepare dataset for Surprise
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

    # Train-test split
    print("Splitting data...")
    trainset, testset = train_test_split(data, test_size=0.2)

    # Train SVD model with reduced complexity
    print("Training model...")
    model = SVD(n_factors=50, n_epochs=10)  # Reduced parameters
    model.fit(trainset)

    # Evaluate model
    print("Evaluating model...")
    predictions = model.test(testset)
    print(f"RMSE: {rmse(predictions)}")

    # Return model and testset for further evaluation
    return model, testset

# Function to evaluate the model
def evaluate_model(model, testset, k=5):
    # Generate predictions for the test set
    predictions = model.test(testset)

    # Calculate Precision@K
    def precision_at_k(predictions, k=5):
        user_predictions = {}
        for uid, _, true_r, est, _ in predictions:
            if uid not in user_predictions:
                user_predictions[uid] = []
            user_predictions[uid].append((true_r, est))

        precisions = []
        for user, preds in user_predictions.items():
            preds.sort(key=lambda x: x[1], reverse=True)
            top_k_preds = preds[:k]
            relevant = [1 if true >= 4 else 0 for true, est in top_k_preds]
            precisions.append(sum(relevant) / k)

        return sum(precisions) / len(precisions)

    precision = precision_at_k(predictions, k)
    print(f"Precision@{k}: {precision}")

    return precision

# Main function for testing
if __name__ == "__main__":
    import pandas as pd

    # Load a smaller subset of data
    ratings = pd.read_csv('ratings.csv').sample(frac=0.1, random_state=42)

    # Train model
    model, testset = train_model(ratings)

    # Evaluate model
    evaluate_model(model, testset, k=5)
