import pandas as pd
import random
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, "data.csv")

# Set random seed for reproducibility
random.seed(42)

products = ["phone", "laptop", "headphones", "watch", "shoes"]

positive_sentences = [
    "I really like this {}",
    "This {} works well",
    "The {} quality is good",
    "Satisfied with this {}",
    "The {} performs nicely",
    "Happy with this {} overall"
]

negative_sentences = [
    "I regret buying this {}",
    "This {} is disappointing",
    "The {} quality is bad",
    "Not happy with this {}",
    "This {} works poorly",
    "The {} is not worth the price"
]

data = []

# Generate positive samples
for i in range(200):
    sentence = random.choice(positive_sentences).format(random.choice(products))
    data.append([sentence + f" {i}", "positive"])

# Generate negative samples
for i in range(200):
    sentence = random.choice(negative_sentences).format(random.choice(products))
    data.append([sentence + f" {i}", "negative"])

# Shuffle the dataset
random.shuffle(data)

# Create DataFrame and save
df = pd.DataFrame(data, columns=["text", "label"])
df.to_csv(FILE_PATH, index=False)

print("✅ 400 samples created successfully!")
print("Saved at:", FILE_PATH)
