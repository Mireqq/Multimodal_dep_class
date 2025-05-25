import torch
import pandas as pd
import concurrent.futures
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from load_models import predict_depression

# -------------------------------
# Step 1: Load and Preprocess Test Dataset
# -------------------------------
def load_test_data(csv_file):
    """Loads the dataset, shuffles it, and selects the first 1000 rows."""
    df = pd.read_csv(csv_file)

    #correct column names
    expected_columns = {"path", "Text", "Severity_Level"}
    if not expected_columns.issubset(df.columns):
        raise ValueError(f"CSV file missing expected columns. Found: {df.columns}")

    # Convert categorical labels to numerical (Non=0, Mild=1, Moderate=2, Severe=3)
    label_mapping = {"non": 0, "mild": 1, "moderate": 2, "severe": 3}
    df["Severity_Level"] = df["Severity_Level"].map(label_mapping)

    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Convert to lists for batch processing
    audio_files = df["path"].tolist()
    text_inputs = df["Text"].tolist()
    true_labels = df["Severity_Level"].tolist()

    return audio_files, text_inputs, true_labels

# -------------------------------
# Step 2: Batch Prediction
# -------------------------------
def predict_batch(audio_files, text_inputs):
    """Predicts depression levels in a batch using GPU-accelerated inference."""
    return predict_depression(audio_files, text_inputs)

# -------------------------------
# Step 3: Run Model on Test Data
# -------------------------------
def evaluate_model(test_csv, batch_size=8, num_workers=4):
    """Evaluates the model using the test dataset with batch processing and GPU acceleration."""
    audio_files, text_inputs, true_labels = load_test_data(test_csv)

    predicted_labels = []
    start_time = time.time()

    print(f"\n Evaluating {len(audio_files)} samples in batches of {batch_size}...\n")

    try:
        torch.cuda.empty_cache()
        batch_size = min(batch_size, 12)
    except:
        batch_size = 8 

    with tqdm(total=len(audio_files), desc="Processing", unit="sample") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []

            # Process data in batches
            for i in range(0, len(audio_files), batch_size):
                batch_audio = audio_files[i : i + batch_size]
                batch_text = text_inputs[i : i + batch_size]
                futures.append(executor.submit(predict_batch, batch_audio, batch_text))

            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                batch_predictions = future.result()
                
                # Convert predictions to class indices
                for pred in batch_predictions:
                    predicted_label = pred[0] if isinstance(pred, list) else pred  
                    predicted_class = ["Non", "Mild", "Moderate", "Severe"].index(predicted_label)
                    predicted_labels.append(predicted_class)

                pbar.update(batch_size)

    # -------------------------------
    # Step 4: Compute Metrics
    # -------------------------------
    accuracy = accuracy_score(true_labels[:len(predicted_labels)], predicted_labels)
    class_report = classification_report(true_labels[:len(predicted_labels)], predicted_labels, target_names=["Non", "Mild", "Moderate", "Severe"])
    conf_matrix = confusion_matrix(true_labels[:len(predicted_labels)], predicted_labels)

    end_time = time.time()
    total_time = end_time - start_time

    print("\n Model Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", class_report)
    print("Confusion Matrix:\n", conf_matrix)
    print(f"Total Evaluation Time: {total_time:.2f} seconds")

    return accuracy, class_report, conf_matrix

# -------------------------------
# Step 5: Run Evaluation
# -------------------------------
if __name__ == "__main__":
    TEST_CSV = "/workspace/merged_dataset_fusion.csv"
    evaluate_model(TEST_CSV, batch_size=8, num_workers=4)
