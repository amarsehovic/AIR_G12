from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from datasets import Dataset
import pandas as pd
import torch

# Load your data
from preprocessing import preprocess_data
from doBert import load_or_generate_movie_embeddings

# Preprocessing
print("Preprocessing data...")
movies, ratings, user_profiles = preprocess_data(testing=True)

# Load or generate embeddings
print("Loading or generating movie embeddings...")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Prepare datasets
def create_dataset(user_profiles, movie_embeddings):
    data = []
    for _, row in user_profiles.iterrows():
        for movie_id in row['movie_list']:
            data.append({
                "input_text": f"User {row['userId']} watched Movie {movie_id}",
                "label": 1  # Placeholder label
            })
    return Dataset.from_pandas(pd.DataFrame(data))

train_dataset = create_dataset(user_profiles.iloc[:int(0.8 * len(user_profiles))], None)
eval_dataset = create_dataset(user_profiles.iloc[int(0.8 * len(user_profiles)):], None)

# Tokenize the datasets
def tokenize_function(examples):
    return tokenizer(examples["input_text"], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# Compute metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
print("Training the model...")
trainer.train()

# Evaluate the model
print("Evaluating the model...")
eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)
