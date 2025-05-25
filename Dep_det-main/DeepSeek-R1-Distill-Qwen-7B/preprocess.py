import pandas as pd
import torch
from transformers import AutoTokenizer
from imblearn.over_sampling import RandomOverSampler
import os

DATA_PATH = "/workspace/cleaned_transcripts_with_labels.csv"
FEATURES_DIR = "/workspace/7bqwen/features"
OUTPUT_FILE = os.path.join(FEATURES_DIR, "deepseek_qwen7b_balanced_dataset.pt")
os.makedirs(FEATURES_DIR, exist_ok=True)

#Load dataset
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

#Drop missing values
df.dropna(subset=["Text", "label"], inplace=True)

#Ensure Text column exists
if "Text" not in df.columns:
    raise ValueError(" Error: 'Text' column missing in dataset!")

label_mapping = {"non": 0, "mild": 1, "moderate": 2, "severe": 3}
df["label"] = df["label"].map(label_mapping)

#label distribution BEFORE oversampling
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

df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

#Load tokenizer
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

#Ensure tokenizer uses correct tokens
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  
    tokenizer.pad_token_id = tokenizer.eos_token_id  

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = 0 

#Debug tokenizer settings
print("\n Tokenizer Details:")
print(f"Vocab size: {tokenizer.vocab_size}")
print(f"Special tokens: {tokenizer.all_special_tokens}")
print(f"Pad token: {tokenizer.pad_token}, ID: {tokenizer.pad_token_id}")
print(f"EOS token: {tokenizer.eos_token}, ID: {tokenizer.eos_token_id}")

#Manual Tokenization Check (Before Batch Processing)
test_text = "uh depression how how it feels"
tokens = tokenizer.tokenize(test_text)

print("\n Manual Tokenization Debugging:")
print(f"Original Text: {test_text}")
print(f"Tokenized Output: {tokens}")
print(f"Token IDs: {tokenizer.convert_tokens_to_ids(tokens)}")

# Apply Chat Template Formatting (if needed)
formatted_text = tokenizer.apply_chat_template([{"role": "user", "content": test_text}], tokenize=False)

tokens_formatted = tokenizer.tokenize(formatted_text)

print("\n Checking Tokenization with Chat Template:")
print(f"Formatted Text: {formatted_text}")
print(f"Formatted Tokenized Output: {tokens_formatted}")
print(f"Formatted Token IDs: {tokenizer.convert_tokens_to_ids(tokens_formatted)}")

#Tokenize dataset using **batch processing**
print("\n Tokenizing dataset...")
formatted_texts = []
for text in df_balanced["Text"].tolist():
    formatted_text = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False)

    # Ensure `<|begin▁of▁sentence|>` appears **exactly once**
    if formatted_text.startswith("<|begin▁of▁sentence|>"):
        formatted_text = formatted_text.lstrip("<|begin▁of▁sentence|>")  # Remove all occurrences at the start

    #Manually add **one** correct instance
    formatted_text = "<|begin▁of▁sentence|>" + formatted_text.strip()

    formatted_texts.append(formatted_text)


tokenized_data = tokenizer.batch_encode_plus(
    formatted_texts,
    padding="longest",
    truncation=True,
    max_length=256,
    return_tensors="pt",
    add_special_tokens=True,
    return_token_type_ids=False
)

#Extract tokenized values
input_ids = tokenized_data["input_ids"]
attention_masks = tokenized_data["attention_mask"]
labels = torch.tensor(df_balanced["label"].values, dtype=torch.long)

#Save processed dataset
torch.save({"input_ids": input_ids, "attention_masks": attention_masks, "labels": labels}, OUTPUT_FILE)

print(f"\n Preprocessed dataset saved at: {OUTPUT_FILE}")

#  Test output correctness (Check First 5 Samples)
print("\n Checking tokenization correctness (First 5 Samples)...\n")
for i in range(5):  # Check first 5 examples
    text_sample = df_balanced.iloc[i]["Text"]
    formatted_text_sample = tokenizer.apply_chat_template([{"role": "user", "content": text_sample}], tokenize=False)

    tokens = tokenizer.encode(formatted_text_sample, add_special_tokens=True)
    decoded_text = tokenizer.decode(tokens)

    print(f" Sample {i+1} Text: {text_sample}")
    print(f" Tokenized Input IDs: {tokens}")
    print(f" Decoded Text: {decoded_text}\n")
