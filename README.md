# EEG Signal Processing and Classification Pipeline

This project implements a complete pipeline for processing EEG signals and training a Vision Transformer (ViT) model for classification. The pipeline consists of two main components:

1. Signal Processing and Spectrogram Generation (`generate_graph.py`)
2. Model Training with Hybrid Loss (`TT_Hybrid_Loss_Cha.py`)

## Project Structure

```
.
├── generate_graph.py      # EEG signal processing and spectrogram generation
├── TT_Hybrid_Loss_Cha.py  # ViT model training with hybrid loss
├── Graph_Test/           # Output directory for processed spectrograms
└── models/               # Output directory for trained models
```

## Prerequisites

- Python 3.7+
- CUDA-capable GPU (recommended)
- Required Python packages:
  - torch
  - numpy
  - scipy
  - wfdb
  - mne
  - ssqueezepy
  - transformers
  - datasets
  - evaluate
  - tqdm
  - scikit-learn

## Installation

1. Clone the repository
2. Install required packages:
```bash
pip install torch numpy scipy wfdb mne ssqueezepy transformers datasets evaluate tqdm scikit-learn
```

## Usage

### 1. Signal Processing and Spectrogram Generation

The `generate_graph.py` script processes EEG signals and generates spectrograms using Continuous Wavelet Transform (CWT) and/or Synchrosqueezing Transform (SSQ).

Configuration (in `generate_graph.py`):
```python
EEG_PATH = Path("/path/to/eeg/data")  # Path to EEG data
OUTPUT_PATH = Path.cwd() / "Graph_Test"  # Output directory
PROCESSES = 10  # Number of parallel processes
MODE = 'cwt'  # Options: 'cwt', 'ssq', 'both'
USE_MNE = False  # Whether to apply MNE ICA denoising
CHANNELS = None  # List channels or None for all
SAMPLE_SEC = 30  # Segment length in seconds
FREQ_RANGE = (0.1, 35)  # Frequency range for spectrograms
IMAGE_SIZE = 384  # Output image size
```

Run the script:
```bash
python generate_graph.py
```

### 2. Model Training

The `TT_Hybrid_Loss_Cha.py` script trains a ViT-Hybrid model on the generated spectrograms using a custom hybrid loss function.

Configuration (in `TT_Hybrid_Loss_Cha.py`):
```python
DS_DIR = Path("./DS")  # Directory containing .arrow files
OUTPUT_ROOT = Path("./models")  # Output directory for models
MODEL_ID = "google/vit-hybrid-base-bit-384"  # Base model
CHANNELS = ["ECG", "C4-M1"]  # Channels to process
SUBFOLDERS = ["CWT", "SSQ"]  # Spectrogram types
```

Run the training:
```bash
python TT_Hybrid_Loss_Cha.py
```

## Features

### Signal Processing
- Optional artifact removal using MNE-ICA
- Generation of CWT and/or SSQ spectrograms
- Support for multiple EEG channels
- Configurable frequency range and image size
- Parallel processing for efficiency

### Model Training
- Vision Transformer (ViT) with hybrid architecture
- Custom hybrid loss function combining:
  - Focal Loss
  - Label Smoothing
  - Adaptive Class Weighting
- Comprehensive metrics tracking:
  - Accuracy, Precision, Recall, F1
  - Cohen's Kappa
  - Balanced Accuracy
  - Hamming Loss
  - Jaccard Score
  - Top-k Accuracy

## Output

### Spectrograms
- Generated in `Graph_Test/<record>/<channel>/{cwt,ssq}/`
- Each spectrogram is saved as a JPEG image
- Labels are saved in `Graph_Test/<record>/labels.txt`

### Models
- Trained models are saved in `models/<channel>_<subfolder>/`
- Checkpoints are saved after each epoch
- Best model is selected based on F1 macro score

## Notes

- The pipeline is designed to handle large EEG datasets efficiently
- GPU acceleration is recommended for model training
- The hybrid loss function helps address class imbalance
- Adaptive class weighting is implemented to improve performance on minority classes

## License

[Specify your license here]

## Citation

If you use this code in your research, please cite:
[Add citation information] 
