import polars as pl
import numpy as np
import random
import os
from pathlib import Path

# ---------------------- Path Settings ---------------------- #
BASE_PATH = Path.cwd()
DS_PATH = BASE_PATH / "DS"
MODEL_OUTPUT_PATH = BASE_PATH / "model_output"

# Set EEG channel name
# chs = ["C4-M1", "C3-M2", "F3-M2", "F4-M1"]
chs = ["C4-M1", "C3-M2", "F3-M2", "F4-M1", "O1-M2", "O2-M1", "E1-M2"]

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# The rocord_files can use the same file for all channels if you set random_seed to the same value in previous code
record_files = np.loadtxt(str(DS_PATH / "cwt" / "test_record_C4-M1.txt"), dtype=str)

ratio = 1
summary_list = []

for ch in chs:

    data = pl.read_csv(str(MODEL_OUTPUT_PATH / ch / f"{ch}_results_mix_pre.csv"))
    random.shuffle(data)
    data = data.sample(fraction=ratio, shuffle=True, seed=RANDOM_SEED)
    save_dir = str(MODEL_OUTPUT_PATH / ch)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    # Compute Top-1, Top-2, and Top-3 accuracies
    top_1_acc = (data["Actual"] == data[f"{ch}Top_1"]).mean()
    top_2_acc = ((data["Actual"] == data[f"{ch}Top_1"]) | 
                (data["Actual"] == data[f"{ch}Top_2"])).mean()
    top_3_acc = ((data["Actual"] == data[f"{ch}Top_1"]) | 
                (data["Actual"] == data[f"{ch}Top_2"]) |
                (data["Actual"] == data[f"{ch}Top_3"])).mean()
    # Display results
    print(f"Top-1 Accuracy: {top_1_acc:.2%}")
    print(f"Top-2 Accuracy: {top_2_acc:.2%}")
    print(f"Top-3 Accuracy: {top_3_acc:.2%}")
    # Save summary to list
    summary_list.append({
        "Channel": ch,
        "Top_1_Accuracy": top_1_acc,
        "Top_2_Accuracy": top_2_acc,
        "Top_3_Accuracy": top_3_acc
    })
    # Add Top-1 to Top-3 match boolean columns
    data = data.with_columns([
        (data["Actual"] == data[f"{ch}Top_1"]).alias("match_top_1"),
        ((data["Actual"] == data[f"{ch}Top_1"]) | (data["Actual"] == data[f"{ch}Top_2"])).alias("match_top_2"),
        ((data["Actual"] == data[f"{ch}Top_1"]) | (data["Actual"] == data[f"{ch}Top_2"]) | (data["Actual"] == data[f"{ch}Top_3"])).alias("match_top_3")
    ])
    # Group by Actual and compute mean of matches
    per_label_accuracy = data.group_by("Actual").agg([
        pl.mean("match_top_1").alias(f"{ch}Top_1_Accuracy"),
        pl.mean("match_top_2").alias(f"{ch}Top_2_Accuracy"),
        pl.mean("match_top_3").alias(f"{ch}Top_3_Accuracy"),
        pl.len().alias("Support")
    ])
    # Save to CSV
    per_label_accuracy_path = str(MODEL_OUTPUT_PATH / ch / f"{ch}_per_label_accuracy.csv")
    per_label_accuracy.write_csv(per_label_accuracy_path)
    print(f"Saving per-label accuracy to {per_label_accuracy_path}")

# Save summary using consistent path
summary_df = pl.DataFrame(summary_list)
summary_path = str(MODEL_OUTPUT_PATH / f"overall_accuracy_summary_{ratio}.csv")
summary_df.write_csv(summary_path)
print(f"Summary saved to: {summary_path}")
