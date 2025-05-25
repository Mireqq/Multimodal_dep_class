import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2Model, AutoTokenizer, AutoModel, AutoConfig
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import torchaudio
from torch.optim import AdamW
import pandas as pd
import logging
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

# Load and Clean Dataset
# -------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

audio_df = pd.read_csv('/workspace/audio_train_dataset.csv', delimiter='\t')
text_df = pd.read_csv('/workspace/cleaned_transcripts_with_labels.csv')

audio_df.columns = audio_df.columns.str.strip()
text_df.columns = text_df.columns.str.strip()

audio_df['Participant_ID'] = audio_df['Participant_ID'].astype(str).str.strip()
text_df['Participant_ID'] = text_df['Participant_ID'].astype(str).str.strip()

label_mapping = {'non': 0, 'mild': 1, 'moderate': 2, 'severe': 3}
audio_df['Severity_Level'] = audio_df['Severity_Level'].map(label_mapping)
text_df['label'] = text_df['label'].map(label_mapping)

audio_df = audio_df[audio_df['Severity_Level'].notna()]
text_df = text_df[text_df['label'].notna()]

combined_df = pd.merge(audio_df, text_df, on='Participant_ID', how='outer')
combined_df = combined_df.dropna(subset=['path', 'Text', 'label'])

def valid_text(x):
    if isinstance(x, list):
        return len(x) > 0
    elif isinstance(x, str):
        return len(x.strip()) > 0  # Check for non-empty string after stripping
    return False

combined_df = combined_df[combined_df['Text'].apply(valid_text)]
combined_df = combined_df.reset_index(drop=True)

logger.info(f"Combined dataset size: {len(combined_df)}")



# Load Pretrained Models
# -------------------------------
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

audio_model = Wav2Vec2Model.from_pretrained("/workspace/Wav2Vec2-Dep-Classification").to(device)
audio_model.eval()
for param in audio_model.parameters():
    param.requires_grad = False

tokenizer = AutoTokenizer.from_pretrained("/workspace/8bllama/models/checkpoint-1755")
text_model = AutoModel.from_pretrained("/workspace/8bllama/models/checkpoint-1755", torch_dtype=torch.bfloat16).to(device)
text_model.eval()
for param in text_model.parameters():
    param.requires_grad = False



# Define Dataset and DataLoader
# -------------------------------

class CombinedDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio_path = self.df.iloc[idx]['path']
        text = self.df.iloc[idx]['Text']
        label = self.df.iloc[idx]['label']

        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            if waveform.ndim == 2:
                waveform = waveform.mean(dim=0, keepdim=True)
            return waveform, text, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            logger.error(f"Error loading audio at {audio_path}: {e}")
            return None, None, None

def custom_collate_fn(batch):
    batch = [item for item in batch if item[0] is not None]
    if batch is None or any(item is None for item in batch):
        return None, None, None


    audio_inputs = [item[0] for item in batch]
    text_inputs = [item[1] for item in batch]
    labels = torch.stack([item[2] for item in batch])

    # --- FIX: Move labels to the device ---
    labels = labels.to(device)

    processed_audio = []
    for audio in audio_inputs:
        if audio.ndim == 2:
            audio = audio.squeeze(0)
        elif audio.ndim == 1:
            pass
        else:
            logger.warning(f"Unexpected audio tensor dimension: {audio.ndim}")
            continue

        target_length = 32000
        if audio.size(-1) < target_length:
            padding = torch.zeros((target_length - audio.size(-1)), dtype=audio.dtype)
            audio = torch.cat([audio, padding], dim=-1)
        elif audio.size(-1) > target_length:
            audio = audio[:target_length]
        processed_audio.append(audio.unsqueeze(0))

    if not processed_audio:
        return None, None, None

    audio_batch = torch.cat(processed_audio, dim=0).to(device)

    tokenizer_output = tokenizer(
        text_inputs,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128
    ).to(device)
    text_batch = tokenizer_output['input_ids']

    return audio_batch, text_batch, labels

# Extract Embeddings
def extract_audio_embeddings(audio_input):
    # Assuming audio_input is already 2D: [batch, samples]
    with torch.no_grad():
        audio_emb = audio_model(audio_input).last_hidden_state
    return audio_emb.float()

def extract_text_embeddings(text_input_ids):
    with torch.no_grad():
        text_emb = text_model(text_input_ids).last_hidden_state
    return text_emb.float()

#Load Pretrained BiLSTM and modify it
class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()
        self.audio_lstm = nn.LSTM(768, 512, batch_first=True, bidirectional=True, num_layers=2)
        # Change the input size of text_lstm to 4096
        self.text_lstm = nn.LSTM(4096, 512, batch_first=True, bidirectional=True, num_layers=2)
        self.audio_norm = nn.LayerNorm(512 * 2)
        self.text_norm = nn.LayerNorm(512 * 2)
        # Adjust fusion_norm and fc input size
        self.fusion_norm = nn.LayerNorm(512 * 4)
        self.cross_attn = nn.MultiheadAttention(embed_dim=1024, num_heads=8, dropout=0.4, batch_first=True)
        self.dropout = nn.Dropout(0.4)
        self.activation = nn.GELU()
        self.fc = nn.Linear(512 * 4, 4)

    def forward(self, audio_emb, text_emb):
        audio_out, _ = self.audio_lstm(audio_emb)
        text_out, _ = self.text_lstm(text_emb)
        fusion_input = torch.cat((audio_out[:, -1, :].unsqueeze(1), text_out[:, -1, :].unsqueeze(1)), dim=1)
        fusion_output, _ = self.cross_attn(fusion_input, fusion_input, fusion_input)
        fusion_output = fusion_output.reshape(fusion_output.shape[0], -1)
        fusion_output = self.fusion_norm(fusion_output)
        fusion_output = self.activation(fusion_output)
        fusion_output = self.dropout(fusion_output)
        output = self.fc(fusion_output)
        return output
model = FusionModel().to(device)
print("Loading checkpoint...")
checkpoint = torch.load("/workspace/bilstm/fusion_model.pth", map_location=device, weights_only=True)
if "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]
model.load_state_dict(checkpoint, strict=False)
print("Checkpoint loaded.")


# Define Loss Function and Optimizer

criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=3e-5)

# Save Checkpoint Function

def save_checkpoint(step, model, optimizer, loss, best_f1, checkpoint_dir, model_name="fusion_model"):
    """Saves a checkpoint and keeps only the two most recent."""
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_step_{step}.pth")
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'best_f1': best_f1
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

    # Keep only the last two checkpoints
    checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith(model_name) and f.endswith(".pth")])
    while len(checkpoints) > 2:
        oldest_checkpoint = os.path.join(checkpoint_dir, checkpoints.pop(0))
        os.remove(oldest_checkpoint)
        print(f"Deleted old checkpoint: {oldest_checkpoint}")

def save_best_model(model, path):
    """Saves the best-performing model."""
    torch.save(model.state_dict(), path)
    print(f"Best model saved at {path}")


# Training Loop, Validation, and Checkpointing

# Split into training, validation and test sets

train_df, temp_df = train_test_split(combined_df, test_size=0.4, random_state=42, stratify=combined_df['label']) # 60/40 split
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label']) # 50/50 split of the 40% (20/20)

train_dataset = CombinedDataset(train_df)
val_dataset = CombinedDataset(val_df)
test_dataset = CombinedDataset(test_df)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=custom_collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=custom_collate_fn) 
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=custom_collate_fn) 

num_epochs = 10
checkpoint_dir = "/workspace/bilstm/checkpoints/"
best_model_path = "/workspace/bilstm/checkpoints/best_fusion_model.pth"
best_f1 = 0.0 
global_step = 0

os.makedirs(checkpoint_dir, exist_ok=True)

# --- Define Checkpoint Saving ---
def save_checkpoint(step, model, optimizer, loss, best_f1, checkpoint_dir, model_name="fusion_model"):
    """Saves a checkpoint and keeps only the two most recent."""
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_step_{step}.pth")
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'best_f1': best_f1
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

    # Keep only the last two checkpoints
    checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith(model_name) and f.endswith(".pth")])
    while len(checkpoints) > 2:
        oldest_checkpoint = os.path.join(checkpoint_dir, checkpoints.pop(0))
        os.remove(oldest_checkpoint)
        print(f"Deleted old checkpoint: {oldest_checkpoint}")

def save_best_model(model, path):
    """Saves the best-performing model."""
    torch.save(model.state_dict(), path)
    print(f"Best model saved at {path}")

# Create checkpoint directory
checkpoint_dir = "/workspace/bilstm/checkpoints"
best_model_path = os.path.join(checkpoint_dir, "best_fusion_model.pth")
os.makedirs(checkpoint_dir, exist_ok=True)

# --- Training Loop ---
best_f1 = 0.0
global_step = 0

for epoch in range(num_epochs):
    print(f"Starting Epoch {epoch + 1}...")
    model.train()
    total_loss = 0
    successful_batches = 0

    for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1} (Train)")):
        if batch is None or any(item is None for item in batch):
            continue

        audio_inputs, text_inputs, labels = batch
        optimizer.zero_grad()
        audio_emb = extract_audio_embeddings(audio_inputs)
        text_emb = extract_text_embeddings(text_inputs)
        outputs = model(audio_emb, text_emb)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        successful_batches += 1
        global_step += 1

    num_batches = successful_batches if successful_batches > 0 else 1
    avg_loss = total_loss / num_batches
    total_loss = 0

    # --- End of Epoch Evaluation ---
    print("Starting Validation...")
    torch.cuda.synchronize()

    model.eval()
    all_val_preds = []
    all_val_labels = []
    total_val_loss = 0
    successful_val_batches = 0

    with torch.no_grad():
        for val_batch in val_dataloader:
            if val_batch is None or any(item is None for item in val_batch):
                continue

            audio_inputs, text_inputs, val_labels = val_batch
            audio_emb = extract_audio_embeddings(audio_inputs)
            text_emb = extract_text_embeddings(text_inputs)
            val_outputs = model(audio_emb, text_emb)
            val_loss = criterion(val_outputs, val_labels)
            total_val_loss += val_loss.item()

            _, val_predicted = torch.max(val_outputs, 1)
            all_val_preds.extend(val_predicted.cpu().numpy())
            all_val_labels.extend(val_labels.cpu().numpy())
            successful_val_batches += 1

    torch.cuda.synchronize()
    print("Finished Validation...")

    num_val_batches = successful_val_batches if successful_val_batches > 0 else 1
    avg_val_loss = total_val_loss / num_val_batches

    if len(all_val_labels) > 0:
        val_accuracy = accuracy_score(all_val_labels, all_val_preds)
        val_f1 = f1_score(all_val_labels, all_val_preds, average='weighted')
        val_precision = precision_score(all_val_labels, all_val_preds, average='weighted', zero_division=1)
        val_recall = recall_score(all_val_labels, all_val_preds, average='weighted', zero_division=1)
        val_confusion_matrix = confusion_matrix(all_val_labels, all_val_preds)
    else:
        val_accuracy = 0
        val_f1 = 0
        val_precision = 0
        val_recall = 0
        val_confusion_matrix = None

    print(f"Epoch {epoch + 1} Summary:")
    print(f"Train Loss: {avg_loss:.4f}")
    print(f"Validation Loss: {avg_val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation F1: {val_f1:.4f}")
    print(f"Validation Precision: {val_precision:.4f}")
    print(f"Validation Recall: {val_recall:.4f}")
    if val_confusion_matrix is not None:
        print(f"Validation Confusion Matrix:\n{val_confusion_matrix}")
    else:
        print("Validation Confusion Matrix: No valid data")
    print("-" * 50)

    # --- Save Checkpoint ---
    save_checkpoint(global_step, model, optimizer, avg_loss, best_f1, checkpoint_dir)

    # --- Save Best Model Based on F1 Score ---
    if val_f1 > best_f1:
        best_f1 = val_f1
        save_best_model(model, best_model_path)

# --- Final Test Evaluation ---
best_model = FusionModel().to(device)
best_model.load_state_dict(torch.load(best_model_path))
best_model.eval()

test_preds = []
test_labels = []
successful_test_batches = 0

with torch.no_grad():
    for batch in test_dataloader:
        if batch is None or any(item is None for item in batch):
            continue
        
        audio_inputs, text_inputs, labels = batch
        audio_emb = extract_audio_embeddings(audio_inputs)
        text_emb = extract_text_embeddings(text_inputs)
        outputs = best_model(audio_emb, text_emb)
        _, predicted = torch.max(outputs, 1)
        test_preds.extend(predicted.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())
        successful_test_batches += 1

if successful_test_batches > 0:
    test_accuracy = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, average='weighted')
    test_precision = precision_score(test_labels, test_preds, average='weighted', zero_division=1)
    test_recall = recall_score(test_labels, test_preds, average='weighted', zero_division=1)
    test_confusion_matrix = confusion_matrix(test_labels, test_preds)
else:
    test_accuracy = 0
    test_f1 = 0
    test_precision = 0
    test_recall = 0
    test_confusion_matrix = None

print("Final Test Results:")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
if test_confusion_matrix is not None:
    print(f"Test Confusion Matrix:\n{test_confusion_matrix}")
else:
    print("Test Confusion Matrix: No valid data")

print("Training and evaluation finished!")