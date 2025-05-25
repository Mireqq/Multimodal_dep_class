import os
import prepare_data
from datasets import load_dataset
import pandas as pd
import build_model
from nested_array_catcher import nested_array_catcher
import torch
import torchaudio
import numpy as np
import librosa
from transformers import AutoConfig, Wav2Vec2Processor, set_seed
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize


# Obtaining data for model testing
def get_test_data(configuration):
    test_filepath = configuration['output_dir'] + "/splits/test.csv"

    if not os.path.exists(test_filepath):
        df = prepare_data.df(configuration['corpora'],
                             configuration['data_path'])
        prepare_data.prepare_splits(df, configuration)

    # Load test data
    test_dataset = load_dataset("csv",
                                data_files={"test": test_filepath},
                                delimiter="\t",
                                cache_dir=configuration['cache_dir']
                                )["test"]

    return test_dataset


# Load Model Configuration
def load_model(configuration, device):
    # Load model configuration, processor, and pretrained checkpoint
    processor_name_or_path = configuration['processor_name_or_path']
    model_name_or_path = os.path.join(configuration['output_dir'], configuration['checkpoint'])

    print('Loading checkoint: ', model_name_or_path)

    config = AutoConfig.from_pretrained(model_name_or_path,
                                        cache_dir=configuration['cache_dir']
                                        )
    processor = Wav2Vec2Processor.from_pretrained(processor_name_or_path,
                                                  cache_dir=configuration['cache_dir']
                                                  )
    model = build_model.Wav2Vec2ForSpeechClassification.from_pretrained(
        model_name_or_path,
        cache_dir=configuration['cache_dir']
    ).to(device)

    return config, processor, model


# Resample the audio files
def speech_file_to_array_fn(batch, processor):
    # The daic-woz dataset audio sampling rate is 16khz
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    speech_array = speech_array.squeeze().numpy()
    speech_array = librosa.resample(y=np.asarray(speech_array), orig_sr=sampling_rate,
                                    target_sr=processor.feature_extractor.sampling_rate)

    speech_array = nested_array_catcher(speech_array)

    batch["speech"] = speech_array
    return batch


# Extract features using the processor
def predict(batch, configuration, processor, model, device):
    features = processor(batch["speech"],
                         sampling_rate=processor.feature_extractor.sampling_rate,
                         return_tensors="pt",
                         padding=True
                         )

    input_values = features.input_values.to(device)

    if configuration['return_attention_mask'] is not False:
        attention_mask = features.attention_mask.to(device)
    else:
        attention_mask = None

    # Pass input values through the model to get predictions
    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits
    torch.cuda.empty_cache()  # Memory freeing

    pred_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy()
    batch["predicted"] = pred_ids
    return batch


def report(configuration, y_true, y_pred, label_names, labels=None):
    # Classification report
    clsf_report = classification_report(
        y_true, y_pred, labels=labels, target_names=label_names, zero_division=0, output_dict=True
    )
    clsf_report_df = pd.DataFrame(clsf_report).transpose()
    print(clsf_report_df)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
    print(cm_df)

    # Calculate MSE and RMSE
    mse = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)

    # Adding MSE and RMSE to the Report
    clsf_report_df['MSE'] = mse
    clsf_report_df['RMSE'] = rmse

    # Convert labels to one-hot encoding for ROC AUC calculation
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3])
    y_pred_bin = label_binarize(y_pred, classes=[0, 1, 2, 3])

    # Compute AUC for each class separately and take the macro average
    roc_auc = roc_auc_score(y_true_bin, y_pred_bin, average="macro")

    print(f"ROC AUC Score (Macro Avg): {roc_auc:.4f}")

    # File save path. If it does not exist, create it
    results_path = os.path.join(configuration['output_dir'], 'results')
    os.makedirs(results_path, exist_ok=True)

    # Plot Multi-Class ROC Curve
    fpr = dict()
    tpr = dict()
    roc_auc_per_class = dict()

    plt.figure(figsize=(8, 6))

    for i in range(len(labels)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
        roc_auc_per_class[i] = auc(fpr[i], tpr[i])

        plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {label_names[i]} (AUC = {roc_auc_per_class[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)  # Random classifier line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-Class ROC Curve')
    plt.legend(loc="lower right")

    # Save ROC curve
    plt.savefig(os.path.join(results_path, 'roc_curve_multiclass.png'))
    plt.close()

    # Save classification report and confusion matrix
    if configuration.get('test_corpora'):
        file_name = f"{configuration['output_dir'].split('/')[-1]}-evaluated-on-{configuration['test_corpora']}_clsf_report.csv"
        cm_file_name = f"{configuration['output_dir'].split('/')[-1]}-evaluated-on-{configuration['test_corpora']}_conf_matrix.csv"
    else:
        file_name = 'clsf_report.csv'
        cm_file_name = 'conf_matrix.csv'

    clsf_report_df.to_csv(os.path.join(results_path, file_name), sep='\t')
    cm_df.to_csv(os.path.join(results_path, cm_file_name), sep='\t')


if __name__ == '__main__':
    configuration = {
        "seed": 42,
        "corpora": "daic_woz",
        "data_path": "/workspace/Depression_Recognition-Code/Preprocessing_code/audio_combine/",
        "output_dir": "/workspace/Depression/models/daic-woz/d2-c2-rmse-roc-mean10-frozen",
        "cache_dir": "/workspace/Depression/content/cache/",
        "processor_name_or_path": "facebook/wav2vec2-base",
        "checkpoint": "checkpoint-3420",
        "return_attention_mask": False,
        "test_corpora": None
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    set_seed(configuration['seed'])
    config, processor, model = load_model(configuration, device)

    test_dataset = get_test_data(configuration)
    test_dataset = test_dataset.map(speech_file_to_array_fn,
                                    fn_kwargs=dict(processor=processor)
                                    )

    result = test_dataset.map(predict,
                              batched=True,
                              batch_size=8,
                              fn_kwargs=dict(configuration=configuration,
                                             processor=processor,
                                             model=model,
                                             device=device
                                             )
                              )

    label_names = [config.id2label[i] for i in range(config.num_labels)]
    labels = list(config.id2label.keys())

    y_true = [config.label2id[name] for name in result["Severity_Level"]]
    y_pred = result["predicted"]

    print("True values: \t", y_true[:5])
    print("Predicted values: \t", y_pred[:5])

    print(classification_report(y_true, y_pred, labels=labels, target_names=label_names))
    report(configuration, y_true, y_pred, label_names, labels)