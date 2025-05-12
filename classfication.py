import torch
import pandas as pd
from transformers import ViTHybridForImageClassification, ViTHybridConfig, ViTHybridImageProcessor
from PIL import Image
import os
from pathlib import Path
import numpy as np
import torch.nn as nn
import polars as pl
import time as t
from tqdm import tqdm
import random

# ---------------------- Path Settings ---------------------- #
BASE_PATH = Path.cwd()
GRAPH_PATH = BASE_PATH / "Graph_Test"
DS_PATH = BASE_PATH / "DS"
MODEL_OUTPUT_PATH = BASE_PATH / "model_output"

# ---------------------- Load Models ---------------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
channels = ["C4-M1", "C3-M2", "F3-M2", "F4-M1", "O1-M2", "O2-M1"]
models_path = [
    # Manually add the best checkpoint
    str(BASE_PATH / "models" / "C4-M1" / "cwt" / "checkpoint-23452"),
    str(BASE_PATH / "models" / "C3-M2" / "cwt" / "checkpoint-23452"),
    str(BASE_PATH / "models" / "F3-M2" / "cwt" / "checkpoint-21320"),
    str(BASE_PATH / "models" / "F4-M1" / "cwt" / "checkpoint-31980"),
    str(BASE_PATH / "models" / "E1-M2" / "cwt" / "checkpoint-25536"),
    str(BASE_PATH / "models" / "O1-M2" / "cwt" / "checkpoint-38304"),
    str(BASE_PATH / "models" / "O2-M1" / "cwt" / "checkpoint-25536")
]

feature_extractor = ViTHybridImageProcessor.from_pretrained("google/vit-hybrid-base-bit-384")

print(f"Using device: {device}")

# ---------------------- Process Images and Inference ---------------------- #
image_path = str(GRAPH_PATH)
record_files = np.loadtxt(str(DS_PATH / "cwt" / "test_record_C4-M1.txt"), dtype=str)

RANDOM_SEED = 42  # Set random seed for reproducibility
random.seed(RANDOM_SEED)  # Set random seed
random.shuffle(record_files)  # Shuffle record list
# Choose number of samples to test
sample = [int(len(record_files))]

MIX_PRE = True  # Whether to use mixed precision
BATCH_SIZE = 128
BATCH_SIZE *= 2 if MIX_PRE else BATCH_SIZE  # Double batch size if using mixed precision

for i in tqdm(range(len(models_path))):
    m = models_path[i]
    channel = channels[i]
    config = ViTHybridConfig.from_pretrained(os.path.join(m, "config.json"))
    model_dyt = ViTHybridForImageClassification.from_pretrained(m, config=config, ignore_mismatched_sizes=True)
    model_dyt.half().to(device) if MIX_PRE else model_dyt.to(device)
    results = []
    for j in tqdm(sample):
        record_file = random.sample(record_files.tolist(), j)
        for record in tqdm(record_file, position=0, leave=False):
            try:
                image_dir = os.path.join(image_path, record, channel, "CWT")
                image_files = sorted(os.listdir(image_dir), key=lambda x: int(x.split('.')[0]))
                labels_raw = np.loadtxt(os.path.join(image_path, record, "label.txt"), dtype=str)
                labels = np.atleast_1d(labels_raw)
            except Exception as e:
                print(f"[WARN] Cannot read record {record} (channel={channel}): {e}")
                continue  # Skip to next record

            batch_images, batch_labels = [], []
            for num, img in enumerate(image_files):
                try:
                    img_path = os.path.join(image_dir, img)
                    image = Image.open(img_path)
                    batch_images.append(image)
                    batch_labels.append(labels[num])
                except Exception as e:
                    print(f"[WARN] Cannot load image {img_path}: {e}")
                    continue  # Skip this image

                # Process batch when full or on last image
                if len(batch_images) == BATCH_SIZE or num == len(image_files) - 1:
                    try:
                        inputs = feature_extractor(images=batch_images, return_tensors="pt").to(device)
                        if MIX_PRE:
                            inputs = {k: v.half() for k, v in inputs.items()}
                        with torch.no_grad():
                            outputs = model_dyt(**inputs)
                    except Exception as e:
                        print(f"[ERROR] Inference failed (record={record}, channel={channel}): {e}")
                        batch_images, batch_labels = [], []
                        continue  # Skip this batch

                    # Get topk and save results
                    logits_batch = outputs.logits
                    for logit, actual_label in zip(logits_batch, batch_labels):
                        topk_values, topk_indices = torch.topk(logit.squeeze(), 3)
                        topk_names = [model_dyt.config.id2label[idx.item()] for idx in topk_indices]
                        results.append({
                            "Actual": actual_label,
                            f"{channel}Top_1": topk_names[0],
                            f"{channel}Top_1_Log": topk_values[0].item(),
                            f"{channel}Top_2": topk_names[1],
                            f"{channel}Top_2_Log": topk_values[1].item(),
                            f"{channel}Top_3": topk_names[2],
                            f"{channel}Top_3_Log": topk_values[2].item(),
                        })
                    batch_images, batch_labels = [], []

        # ---------------------- Create DataFrame ---------------------- #
        df = pl.DataFrame(results)
        # ---------------------- Save Results ---------------------- #
        save_dir = str(MODEL_OUTPUT_PATH)
        os.makedirs(os.path.join(save_dir, channel), exist_ok=True)
        df_path = os.path.join(save_dir, channel, f"{channel}_results_mix_pre.csv")
        df.write_csv(df_path)
        print(f"Results saved to {df_path}")