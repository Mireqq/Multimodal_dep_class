import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torchaudio
from sklearn.model_selection import train_test_split
import os
from sklearn.utils import resample

config = {
    "seed": 42,
    "corpora": "daic_woz",
    "data_path": "/workspace/Depression_Recognition-Code/Preprocessing_code/audio_combine/",
    "output_dir": "/workspace/Depression/models/daic-woz/d2-c2-rmse-roc-mean10-frozen"
}

# Load PHQ-8 Data
combined_data_path = "/workspace/Depression/Model_training_code/combined_depression_data.csv"
combined_df = pd.read_csv(combined_data_path)
combined_df["Participant_ID"] = combined_df["Participant_ID"].astype(str)

# Target samples per class
TARGET_SAMPLES = 2000  # Balancing target


# Used to organise and create a unified dataframe for all datasets.
class CorpusDataFrame():
    def __init__(self):
        self.data = []
        self.exceptions = 0

    def append_file(self, path, name, participant_id, phq_score, phq_binary, severity_label):
        # Append filename, filepath, and severity label to the data list.
        try:
            # Avoid broken files
            s = torchaudio.load(path)
            self.data.append({
                "name": name,
                "path": path,
                "Participant_ID": participant_id,
                "PHQ-8": phq_score,
                "PHQ8_Binary": phq_binary,
                "Severity_Level": severity_label
            })
        except Exception as e:
            print(f'Could not load {str(path)}', e)
            self.exceptions += 1
            pass

    def data_frame(self):
        if self.exceptions > 0:
            print(f'{self.exceptions} files could not be loaded')
        # Create the dataframe from the organised data list
        df = pd.DataFrame(self.data)
        return df


# Function to create the DAIC_WOZ dataset
def DAIC_WOZ(data_path):
    print('PREPARING DAIC_WOZ DATA PATHS')

    cdf = CorpusDataFrame()

    for path in tqdm(Path(data_path).glob("**/*.wav")):
        filename = str(path).split('/')[-1]  # Get file name
        participant_id = filename.split('_')[0]  # Extract participant ID

        # Find corresponding PHQ-8 score from the CSV file
        participant_data = combined_df[combined_df["Participant_ID"] == participant_id]

        if participant_data.empty:
            print(f"No PHQ-8 data found for Participant {participant_id}, skipping...")
            continue

        phq_score = int(participant_data["PHQ-8"].values[0])
        phq_binary = int(participant_data["PHQ8_Binary"].values[0])

        # Assign severity levels based on PHQ-8
        if 0 <= phq_score <= 4:
            severity_label = 'non'
        elif 5 <= phq_score <= 9:
            severity_label = 'mild'
        elif 10 <= phq_score <= 14:
            severity_label = 'moderate'
        else:
            severity_label = 'severe'

        # Append to dataset
        cdf.append_file(path, filename, participant_id, phq_score, phq_binary, severity_label)

    df = cdf.data_frame()
    return df


# Function to balance dataset
def balance_classes(df):
    class_counts = df["Severity_Level"].value_counts()
    print("Original Class Distribution:\n", class_counts)

    balanced_df = pd.DataFrame()

    for class_label in class_counts.index:
        class_subset = df[df["Severity_Level"] == class_label]

        if len(class_subset) > TARGET_SAMPLES:
            # Undersample majority class
            balanced_subset = class_subset.sample(TARGET_SAMPLES, random_state=42)
        else:
            # Oversample minority class
            balanced_subset = resample(class_subset, replace=True, n_samples=TARGET_SAMPLES, random_state=42)

        balanced_df = pd.concat([balanced_df, balanced_subset])

    # Shuffle the final dataset
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("Balanced Class Distribution:\n", balanced_df["Severity_Level"].value_counts())
    return balanced_df

def get_df(corpus, data_path):
    if corpus == 'daic_woz':
        df = DAIC_WOZ(data_path)
    else:
        raise ValueError("Invalid corpus name")

    return df


# To get the dataset and balance classes
def prepare_df(corpora, data_path):
    df = get_df(corpora, data_path)

    print(f"Step 0: {len(df)}")

    # Filter out non-existing files
    df["status"] = df["path"].apply(lambda path: True if os.path.exists(path) else None)
    df = df.dropna(subset=["path"])
    df = df.drop("status", axis=1)
    print(f"Step 1: {len(df)}")

    # Balance classes
    df = balance_classes(df)

    print("Labels: ", df["Severity_Level"].unique())
    df.groupby("Severity_Level").count()[["path"]]

    return df


# Function to create splits (80% train, 10% validation, 10% test)
def prepare_splits(df, config):
    output_dir = config['output_dir']
    save_path = output_dir + "/splits/"

    # Create splits directory
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # Define proportions
    random_state = config['seed']
    train_df, temp_df = train_test_split(df, test_size=0.2, train_size=0.8, random_state=random_state,
                                         stratify=df["Severity_Level"])
    valid_df, test_df = train_test_split(temp_df, test_size=0.5, train_size=0.5, random_state=random_state,
                                         stratify=temp_df["Severity_Level"])

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)

    # Save each to file
    train_df.to_csv(f"{save_path}/train.csv", sep="\t", encoding="utf-8", index=False)
    test_df.to_csv(f"{save_path}/test.csv", sep="\t", encoding="utf-8", index=False)
    valid_df.to_csv(f"{save_path}/valid.csv", sep="\t", encoding="utf-8", index=False)

    print(f'Train: {train_df.shape}, Validate: {valid_df.shape}, Test: {test_df.shape}')


if __name__ == '__main__':
    train_filepath = config['output_dir'] + "/splits/train.csv"
    test_filepath = config['output_dir'] + "/splits/test.csv"
    valid_filepath = config['output_dir'] + "/splits/valid.csv"

    df = prepare_df(config['corpora'], config['data_path'])
    prepare_splits(df, config)
