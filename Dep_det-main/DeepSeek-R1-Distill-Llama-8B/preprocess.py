import pandas as pd
import torch
from transformers import AutoTokenizer
from imblearn.over_sampling import RandomOverSampler
import os

DATA_PATH = "/workspace/cleaned_transcripts_with_labels.csv"
FEATURES_DIR = "/workspace/8bllama"
OUTPUT_FILE = os.path.join(FEATURES_DIR, "deepseek_llama_balanced_dataset.pt")
os.makedirs(FEATURES_DIR, exist_ok=True)

# Load dataset
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

#Fix Column Names if Needed
df.rename(columns={"answer": "Text"}, inplace=True)
if "Text" not in df.columns:
    raise ValueError("Error: 'Text' column missing in dataset!")

# Drop missing values
df.dropna(subset=["Text", "label"], inplace=True)

label_mapping = {"non": 0, "mild": 1, "moderate": 2, "severe": 3}
df["label"] = df["label"].map(label_mapping)

# label distribution BEFORE oversampling
print("\n Label distribution BEFORE oversampling:")
print(df["label"].value_counts())

# Balance dataset using Random Over-Sampling
oversampler = RandomOverSampler(sampling_strategy="auto", random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(df[["Text"]], df["label"])

# Convert resampled data into DataFrame
df_balanced = pd.DataFrame({"Text": X_resampled["Text"].values, "label": y_resampled})

# Show label distribution AFTER oversampling
print("\n Label distribution AFTER oversampling:")
print(df_balanced["label"].value_counts())

# Shuffle dataset
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

#Load **DeepSeek-LLaMA-8B** tokenizer
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

#Fix missing pad token (LLaMA models don’t always have one)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Set pad_token to eos_token
tokenizer.padding_side = "right"  # Ensures correct padding behavior

#Tokenization process (No Chat Formatting)
print("\n Tokenizing dataset...")
tokenized_texts = tokenizer(
    df_balanced["Text"].tolist(),
    padding="max_length",
    truncation=True,
    max_length=256,  # Adjust if needed
    return_tensors="pt"
)

#Convert labels to tensor
labels = torch.tensor(df_balanced["label"].values, dtype=torch.long)

#Save preprocessed dataset
torch.save(
    {
        "input_ids": tokenized_texts["input_ids"],
        "attention_masks": tokenized_texts["attention_mask"],
        "labels": labels,
    },
    OUTPUT_FILE,
)

print(f"\n Preprocessed dataset saved at: {OUTPUT_FILE}")
print("\n Checking tokenization correctness...\n")

sample_index = 5 
sample_text = df_balanced.iloc[sample_index]["Text"]
tokenized_sample = tokenizer(sample_text, padding=True, truncation=True, max_length=256)

decoded_text = tokenizer.decode(tokenized_sample["input_ids"])

print(f"Sample Input Text:\n{sample_text}\n")
print(f"Tokenized Input IDs:\n{tokenized_sample['input_ids']}\n")
print(f"Decoded Text:\n{decoded_text}\n")
