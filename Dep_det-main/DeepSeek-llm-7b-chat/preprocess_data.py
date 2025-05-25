import pandas as pd
import torch
from transformers import AutoTokenizer
from imblearn.over_sampling import RandomOverSampler
import os

DATA_PATH = "/workspace/cleaned_transcripts_with_labels.csv"
FEATURES_DIR = "/workspace/features"
OUTPUT_FILE = os.path.join(FEATURES_DIR, "deepseek_balanced_dataset.pt")
os.makedirs(FEATURES_DIR, exist_ok=True)

#Load dataset
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

#Drop missing values
df.dropna(subset=["Text", "label"], inplace=True)

#Ensure Text column exists
if "Text" not in df.columns:
    raise ValueError("Error: 'Text' column missing in dataset!")

label_mapping = {"non": 0, "mild": 1, "moderate": 2, "severe": 3}
df["label"] = df["label"].map(label_mapping)

#Show label distribution BEFORE oversampling
print("\n Label distribution BEFORE oversampling:")
print(df["label"].value_counts())

#Balance dataset using Random Over-Sampling
oversampler = RandomOverSampler(sampling_strategy="auto", random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(df[["Text"]], df["label"])

#Convert resampled data into DataFrame
df_balanced = pd.DataFrame({"Text": X_resampled["Text"].values, "label": y_resampled})

#Show label distribution AFTER oversampling
print("\n Label distribution AFTER oversampling:")
print(df_balanced["label"].value_counts())

#Shuffle dataset
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

#Load **DeepSeek 7B Chat** tokenizer
MODEL_NAME = "deepseek-ai/deepseek-llm-7b-chat"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

#Tokenize text using **apply_chat_template()**
print("\n Tokenizing dataset...")
formatted_inputs = [
    tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=False)
    for text in df_balanced["Text"]
]

#Padding and truncation for consistency
max_length = 256  # Adjust as needed
input_ids = torch.full((len(formatted_inputs), max_length), tokenizer.pad_token_id, dtype=torch.long)
attention_masks = torch.zeros((len(formatted_inputs), max_length), dtype=torch.long)

for i, tokens in enumerate(formatted_inputs):
    tokens = tokens[:max_length]  # Truncate if too long
    input_ids[i, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)  # Fill in the values
    attention_masks[i, : len(tokens)] = 1  # Mark actual tokens

labels = torch.tensor(df_balanced["label"].values)

#Save preprocessed dataset
torch.save({"input_ids": input_ids, "attention_masks": attention_masks, "labels": labels}, OUTPUT_FILE)

print(f"\n Preprocessed dataset saved at: {OUTPUT_FILE}")

#Test output correctness
print("\n Checking tokenization correctness...\n")

# Select a **random** sample to verify the tokenization
sample_index = 5  # Change index to inspect different samples
sample_text = df_balanced.iloc[sample_index]["Text"]
tokenized_sample = tokenizer.apply_chat_template(
    [{"role": "user", "content": sample_text}], tokenize=True, add_generation_prompt=False
)
decoded_text = tokenizer.decode(tokenized_sample)

print(f"Sample Input Text:\n{sample_text}\n")
print(f"Tokenized Input IDs:\n{tokenized_sample}\n")
print(f"Decoded Text:\n{decoded_text}\n")
