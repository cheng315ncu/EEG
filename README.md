# EEG Sleep Stage Classification using ViT-Hybrid

This project implements a deep learning-based approach for EEG sleep stage classification using the Vision Transformer (ViT) Hybrid architecture. The model is fine-tuned on spectrogram representations of EEG signals.

## Dataset

The project uses the [PhysioNet Challenge 2017](https://physionet.org/content/challenge-2017/1.0.0/) dataset for training and evaluation. The dataset contains EEG recordings with sleep stage annotations.

## Model Architecture

The project uses the [google/vit-hybrid-base-bit-384](https://huggingface.co/google/vit-hybrid-base-bit-384) model from Hugging Face, which is a Vision Transformer (ViT) hybrid architecture pre-trained on ImageNet and fine-tuned for our specific task.

## Project Structure

```
BASE_PATH/
├── Graph_Test/                    # Generated spectrograms
│   ├── [record_name]/
│   │   ├── [channel]/
│   │   │   ├── cwt/              # Continuous Wavelet Transform spectrograms
│   │   │   └── ssq/              # Synchrosqueezing Transform spectrograms
│   │   └── labels.txt            # Sleep stage labels
├── DS/                           # Processed datasets
│   ├── cwt/
│   │   ├── dataset_[channel]_train.arrow
│   │   └── dataset_[channel]_test.arrow
│   └── test_record_[channel].txt
├── models/                       # Model checkpoints
│   ├── [channel]/
│   │   └── cwt/
│   │       └── checkpoint-[id]/
├── model_output/                 # Inference results
│   ├── [channel]/
│   │   ├── [channel]_results_mix_pre.csv
│   │   ├── [channel]_per_label_accuracy.csv
│   │   └── [channel]_confusion_matrix.csv
│   └── overall_accuracy_summary.csv
└──                              # Source code
    ├── generate_graph.py        # Generate spectrograms
    ├── datasets_make.py         # Create datasets
    ├── TT_Hybrid_Loss_Cha.py    # Model training
    ├── classification.py        # Model inference
    ├── evaluate.py              # Evaluation metrics
    └── confusion_matrix.py      # Confusion matrix generation
```

## Dependencies

```python
torch
transformers
polars
numpy
scipy
wfdb
mne
ssqueezepy
PIL
tqdm
sklearn
```

## Processing Pipeline

1. **Data Preprocessing** (`generate_graph.py`):
   - Reads raw EEG data
   - Generates CWT and SSQ spectrograms
   - Saves spectrograms and labels

2. **Dataset Creation** (`datasets_make.py`):
   - Processes spectrograms into training and test sets
   - Creates Arrow datasets for efficient loading

3. **Model Training** (`TT_Hybrid_Loss_Cha.py`):
   - Fine-tunes ViT-Hybrid model
   - Uses focal loss with label smoothing
   - Implements adaptive weighting

4. **Inference** (`classification.py`):
   - Loads trained models
   - Performs batch inference
   - Saves prediction results

5. **Evaluation** (`evaluate.py`, `confusion_matrix.py`):
   - Calculates accuracy metrics
   - Generates confusion matrices
   - Produces per-label accuracy reports

## Usage

1. **Generate Spectrograms**:
```bash
python generate_graph.py
```

2. **Create Datasets**:
```bash
python datasets_make.py
```

3. **Train Models**:
```bash
python TT_Hybrid_Loss_Cha.py
```

4. **Run Inference**:
```bash
python classification.py
```

5. **Evaluate Results**:
```bash
python evaluate.py
python confusion_matrix.py
```

## Model Performance

The model is evaluated using:
- Top-1, Top-2, and Top-3 accuracy
- Per-label accuracy
- Confusion matrix analysis

Results are saved in the `model_output` directory with detailed metrics for each channel.

## Citation

If you use this code, please cite:
1. The PhysioNet Challenge 2017 dataset
2. The ViT-Hybrid model from Hugging Face
3. This repository

## License

[Add your license information here]

## Acknowledgments

- PhysioNet for providing the dataset
- Hugging Face for the pre-trained model
