"""
datasets_make.py

Automatically generate Arrow datasets from EEG Graphs:
- Process CWT and/or SSQ spectrograms per channel
- Save outputs under ./DS/<target>/<channel>/
- Save labels per record in ./DS/<target>/<channel>/labels.txt
- Save test set records in ./DS/<target>/<channel>_test_records.txt

Configuration is embedded; adjust EEG_PATH as needed.
"""
import logging
from pathlib import Path
import numpy as np
from datasets import Dataset, Features, ClassLabel, Image
import random
import os
from tqdm import tqdm
import polars as pl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dataset_generation.log')
    ]
)
logger = logging.getLogger(__name__)

""" CHANNELS can choose from:
    'SaO2': 'misc', 
    'ABD': 'emg', 
    'CHEST': 'emg', 
    'Chin1-Chin2': 'emg',
    'AIRFLOW': 'misc', 
    'ECG': 'ecg', 
    'E1-M2': 'eog
    'C4-M1': 'eeg', 
    'C3-M2': 'eeg', 
    'F3-M2': 'eeg', 
    'F4-M1': 'eeg',
    'O1-M2': 'eeg',
    'O2-M1': 'eeg'
}
"""	

Graph_Path = Path.cwd() / "Graph_Test"
CHANNELS = ["C4-M1"]
# Set split ratio
RATIO_TRAIN = 0.85
RANDOM_SEED = 42 # Set random seed for reproducibility

# --- Get and shuffle record list ---
record_list = os.listdir(Graph_Path)
random.seed(RANDOM_SEED) # Set random seed
random.shuffle(record_list) # *** Shuffle record list ***
num_total = len(record_list)

logger.info(f"Total records: {num_total}")

# Calculate number of training records based on ratio
num_train_records = int(num_total * RATIO_TRAIN)
num_test_records = num_total - num_train_records

logger.info(f"Splitting records: {num_train_records} for training, {num_test_records} for testing.")

# --- Define dataset structure ---
class_labels = ClassLabel(names=['W', 'R', 'N1', 'N2', 'N3'])
features = Features({
    "image": Image(),
    "labels": class_labels
})

def makedatasets(save_test_ds: bool = False, Target: list[str] = ["cwt", "ssq"]):
    """
    Generate datasets for specified targets (CWT and/or SSQ).
    
    Args:
        save_test_ds (bool): Whether to save test datasets
        Target (list[str]): List of targets to process (e.g., ["cwt", "ssq"])
    """
    # --- Start processing each channel ---
    for i in tqdm(CHANNELS, desc=f"      Processing Channels", ncols=100, ascii="░▒█", position=0, leave=True):
        for target in Target:
            train_df_list = [] # List to store training data DataFrames
            test_df_list = []  # List to store testing data DataFrames
            test_record = []   # List to store test record names

            logger.info(f"\n--- Channel: {i}, Target: {target} ---")

            # --- Phase 1: Process training records ---
            # Use first num_train_records from shuffled list
            for j in tqdm(range(num_train_records), desc=f"  Processing Train Records", ncols=100, ascii="░▒█", position=1, leave=False):
                record_name = record_list[j]
                path = os.path.join(Graph_Path, record_name)
                image_path = os.path.join(path, i, target)

                if not os.path.exists(image_path):
                    logger.warning(f"Train path {image_path} does not exist. Skipping...")
                    continue

                images = sorted(os.listdir(image_path), key=lambda x: int(x.split('.')[0]))
                images = [os.path.join(image_path, image) for image in images]

                try:
                    label = np.loadtxt(os.path.join(path, "label.txt"), delimiter=',', dtype=str)
                    if label.ndim == 0:
                        label = np.array([label])
                    label = [l.strip() for l in label]
                except Exception as e:
                    logger.error(f"Error loading labels from train record {path}: {e}. Skipping...")
                    continue

                if len(images) != len(label):
                    logger.warning(f"Mismatch in train record {path} (Images: {len(images)}, Labels: {len(label)}). Skipping...")
                    continue

                if images: # Ensure list is not empty
                    df_temp = pl.DataFrame({"image": images, "labels": label})
                    train_df_list.append(df_temp)

            # --- Phase 2: Process testing records ---
            # Use remaining records from shuffled list
            for k in tqdm(range(num_train_records, num_total), desc=f"  Processing Test Records ", ncols=100, ascii="░▒█", position=1, leave=False):
                record_name = record_list[k]
                test_record.append(record_name)
                path = os.path.join(Graph_Path, record_name)
                image_path = os.path.join(path, i, target)

                if not os.path.exists(image_path):
                    logger.warning(f"Test path {image_path} does not exist. Skipping...")
                    continue

                images = sorted(os.listdir(image_path), key=lambda x: int(x.split('.')[0]))
                images = [os.path.join(image_path, image) for image in images]

                try:
                    label = np.loadtxt(os.path.join(path, "label.txt"), delimiter=',', dtype=str)
                    if label.ndim == 0:
                        label = np.array([label])
                    label = [l.strip() for l in label]
                except Exception as e:
                    logger.error(f"Error loading labels from test record {path}: {e}. Skipping...")
                    continue

                if len(images) != len(label):
                    logger.warning(f"Mismatch in test record {path} (Images: {len(images)}, Labels: {len(label)}). Skipping...")
                    continue

                # *** Note: No longer filtering R and N3 labels, including all samples in test records ***
                if images: # Ensure list is not empty
                    df_temp = pl.DataFrame({"image": images, "labels": label})
                    test_df_list.append(df_temp)

            # --- Phase 3: Merge, transform and create training dataset ---
            train_ds = None
            if train_df_list:
                train_df = pl.concat(train_df_list)
                logger.info("\nTraining data distribution:")
                logger.info(train_df.group_by("labels").agg(pl.len().alias("count")).sort("labels"))
                logger.info(f"Total training samples: {len(train_df)}")

                train_pandas = train_df.to_pandas()
                train_pandas["image"] = train_pandas["image"].apply(lambda x: {"path": x})
                current_features = Features({col: features[col] for col in train_pandas.columns})
                try:
                    train_ds = Dataset.from_pandas(train_pandas, features=current_features)
                except Exception as e:
                    logger.error(f"Error creating training Dataset for channel {i}: {e}")
                    train_ds = None # Mark as None
            else:
                logger.warning(f"\nNo valid training data collected for channel {i}.")
                # Create empty training set for consistency
                train_ds = Dataset.from_dict({k: [] for k in features.keys()}, features=features)

            # --- Phase 4: Merge, transform and create testing dataset ---
            test_ds = None
            if test_df_list:
                test_df = pl.concat(test_df_list)
                logger.info("\nTesting data distribution:")
                logger.info(test_df.group_by("labels").agg(pl.len().alias("count")).sort("labels"))
                logger.info(f"Total testing samples: {len(test_df)}")

                test_pandas = test_df.to_pandas()
                test_pandas["image"] = test_pandas["image"].apply(lambda x: {"path": x})
                current_features = Features({col: features[col] for col in test_pandas.columns})
                try:
                    test_ds = Dataset.from_pandas(test_pandas, features=current_features)
                except Exception as e:
                    logger.error(f"Error creating testing Dataset for channel {i}: {e}")
                    test_ds = None # Mark as None
            else:
                logger.warning(f"\nNo valid testing data collected for channel {i}.")
                # Create empty test set
                test_ds = Dataset.from_dict({k: [] for k in features.keys()}, features=features)

            # --- Phase 5: Save split datasets ---
            base_save_path = Path.cwd() / "DS" / target # Base save path
            os.makedirs(base_save_path, exist_ok=True) # Ensure directory exists

            train_save_path = os.path.join(base_save_path, f'dataset_{i}_train.arrow')
            test_save_path = os.path.join(base_save_path, f'dataset_{i}_test.arrow')

            # Check if Dataset was created successfully before saving
            if train_ds is not None:
                try:
                    train_ds.save_to_disk(train_save_path)
                    logger.info(f"Training dataset saved to {train_save_path}")
                except Exception as e:
                    logger.error(f"Error saving training dataset for channel {i}: {e}")
            else:
                logger.warning("Skipping saving empty or failed training dataset.")

            if test_ds is not None:
                try:
                    if save_test_ds:
                        test_ds.save_to_disk(test_save_path)
                    logger.info(f"Testing dataset record saved to {test_save_path}")
                    np.savetxt(os.path.join(base_save_path, f'test_record_{i}.txt'), test_record, fmt="%s")
                except Exception as e:
                    logger.error(f"Error saving testing dataset for channel {i}: {e}")
            else:
                logger.warning("Skipping saving empty or failed testing dataset.")

    logger.info("\nProcessing Finished.")

if __name__ == "__main__":
    makedatasets(save_test_ds=False)
