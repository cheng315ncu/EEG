"""
datasets_make.py

Automatically generate Arrow datasets from EEG Graphs:
- Process CWT and/or SSQ spectrograms per channel
- Save outputs under ./DS/<channel>/{cwt,ssq}/
- Save labels per record in ./DS/<channel>/labels.txt

Configuration is embedded; adjust EEG_PATH as needed.
"""
import logging
from pathlib import Path

import numpy as np
from datasets import Dataset, Features, ClassLabel, Image

# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
GRAPH_PATH = SCRIPT_DIR / "Graph"
OUT_DIR = Path.cwd() / "DS"

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
CHANNELS = ["C4-M1"]
# Choose subfolder to process: "CWT", "SSQ", or "both"
PROCESS_SUBFOLDER = "both"

# 70% for training, 15% for testing, 15% for validation
TRAIN_RATIO = 0.85
SEED = 42
CLASS_NAMES = ["W", "R", "N1", "N2", "N3"]

# --- Logging configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)

def get_records(graph_path: Path):
    """Return a sorted list of all record directories under graph_path."""
    return sorted([p for p in graph_path.iterdir() if p.is_dir()])

def load_samples(record_path: Path, channel: str, subfolder: str):
    """
    Load samples for a given record/channel/subfolder.
    Returns a list of dicts: [{'image': {'path': ...}, 'labels': ...}, ...].
    If the folder does not exist or image/label count mismatch, returns an empty list.
    """
    data_dir = record_path / channel / subfolder
    if not data_dir.exists():
        return []
    images = sorted(data_dir.iterdir(), key=lambda p: int(p.stem))
    try:
        raw = np.loadtxt(record_path / "label.txt", delimiter=",", dtype=str)
    except Exception as e:
        logging.warning(f"Failed to read {record_path}/label.txt: {e}")
        return []
    labels = raw.reshape(-1)
    if len(images) != len(labels):
        logging.warning(
            f"{record_path.name}/{channel}/{subfolder} images ({len(images)}) "
            f"and labels ({len(labels)}) count mismatch, skipping."
        )
        return []
    return [
        {"image": {"path": str(img)}, "labels": lab.strip()}
        for img, lab in zip(images, labels)
    ]

def process_channel(
    graph_path: Path,
    channel: str,
    subfolder: str,
    out_dir: Path,
    train_ratio: float,
    seed: int,
    class_names: list[str]
):
    """
    Load all samples for channel/subfolder, split into train/test,
    and save as Arrow datasets.
    """
    records = get_records(graph_path)
    samples = []
    for rec in records:
        samples.extend(load_samples(rec, channel, subfolder))

    if not samples:
        logging.warning(f"No valid samples for channel {channel}/{subfolder}, skipping.")
        return

    features = Features({
        "image": Image(),
        "labels": ClassLabel(names=class_names)
    })
    ds = Dataset.from_list(samples, features=features)
    ds = ds.train_test_split(train_size=train_ratio, seed=seed)
    train_ds, test_ds = ds["train"], ds["test"]

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / f"{channel}_{subfolder}_train.arrow"
    test_path = out_dir / f"{channel}_{subfolder}_test.arrow"

    train_ds.save_to_disk(str(train_path))
    test_ds.save_to_disk(str(test_path))
    logging.info(
        f"[{channel}/{subfolder}] Saved train={len(train_ds)} to {train_path}, "
        f"test={len(test_ds)} to {test_path}"
    )

if __name__ == "__main__":
    logging.info(f"SCRIPT_DIR = {SCRIPT_DIR}")
    logging.info(f"GRAPH_PATH = {GRAPH_PATH}")
    logging.info(f"OUT_DIR = {OUT_DIR}")
    logging.info(f"PROCESS_SUBFOLDER = {PROCESS_SUBFOLDER}")

    # Determine subfolders to process based on setting
    if PROCESS_SUBFOLDER.lower() == "both":
        subfolders = ["CWT", "SSQ"]
    else:
        subfolders = [PROCESS_SUBFOLDER]

    for ch in CHANNELS:
        for sub in subfolders:
            logging.info(f"Processing {ch}/{sub} ...")
            process_channel(
                graph_path=GRAPH_PATH,
                channel=ch,
                subfolder=sub,
                out_dir=OUT_DIR,
                train_ratio=TRAIN_RATIO,
                seed=SEED,
                class_names=CLASS_NAMES
            )
    logging.info("All channels processed.")
