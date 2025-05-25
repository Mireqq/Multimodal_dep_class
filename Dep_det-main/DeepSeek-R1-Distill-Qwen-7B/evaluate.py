import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix, 
    mean_squared_error, roc_auc_score, roc_curve, auc
)

MODEL_PATH = "/workspace/7bqwen/models"
DATA_PATH = "/workspace/7bqwen/deepseek_balanced_dataset.pt"
CACHE_DIR = "/workspace/deepseek_env"

#Load Model & Tokenizer
print("Loading trained model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, cache_dir=CACHE_DIR).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, cache_dir=CACHE_DIR)
model.eval()

#Load Evaluation Data
print("Loading dataset...")
data = torch.load(DATA_PATH)
input_ids = torch.tensor(data["input_ids"], dtype=torch.long)
attention_masks = torch.tensor(data["attention_masks"], dtype=torch.long)
labels = torch.tensor(data["labels"], dtype=torch.long)

#Convert to Hugging Face Dataset format
eval_dataset = Dataset.from_dict({
    "input_ids": input_ids.tolist(),
    "attention_mask": attention_masks.tolist(),
    "labels": labels.tolist()
})

#DataLoader for batch processing
batch_size = 64
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)

#Evaluation Function (Fixed `bfloat16` Issue)
def evaluate(model, dataloader):
    all_predictions, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
            # Convert batch list into dictionary of tensors
            batch_dict = {key: torch.tensor([item[key] for item in batch]).to(device) for key in batch[0]}

            #Extract input tensors
            input_ids = batch_dict["input_ids"]
            attention_masks = batch_dict["attention_mask"]
            labels = batch_dict["labels"]

            #Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_masks)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits.to(torch.float32), dim=1).cpu().numpy()
            probabilities /= probabilities.sum(axis=1, keepdims=True)

            predictions = torch.argmax(logits, dim=1)

            #Convert `bfloat16` to `float32` before `.numpy()`
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities.astype(np.float32))

    #Compute Metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average=None, zero_division=1)

    #MSE & RMSE
    mse = mean_squared_error(all_labels, all_predictions)
    rmse = np.sqrt(mse)

    #ROC AUC Score (Macro Average)
    try:
        roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    except ValueError:
        roc_auc = None

    #Confusion Matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    return {
        "accuracy": accuracy,
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1_score": f1.tolist(),
        "mse": mse,
        "rmse": rmse,
        "roc_auc": roc_auc,
        "conf_matrix": conf_matrix
    }, all_labels, all_probs

#Run Evaluation
print(Evaluating model...")
metrics, y_true, y_probs = evaluate(model, eval_dataloader)

#Print Metrics
print(f"\n Evaluation Results:\n")
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']}")
print(f"Recall: {metrics['recall']}")
print(f"F1-score: {metrics['f1_score']}")
print(f"MSE: {metrics['mse']:.4f}")
print(f"RMSE: {metrics['rmse']:.4f}")
print(f"ROC AUC Score: {metrics['roc_auc']:.4f}" if metrics['roc_auc'] else "ROC AUC Score: Not Computable")

#Save Confusion Matrix Plot
plt.figure(figsize=(8, 6))
sns.heatmap(metrics["conf_matrix"], annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Non", "Mild", "Moderate", "Severe"], 
            yticklabels=["Non", "Mild", "Moderate", "Severe"])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()

#Save ROC Curve
if metrics["roc_auc"]:
    plt.figure(figsize=(8, 6))
    for i in range(4):  # 4 classes: "non", "mild", "moderate", "severe"
        fpr, tpr, _ = roc_curve((np.array(y_true) == i).astype(int), np.array(y_probs)[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-Class ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig("roc_curve.png")
    plt.close()
