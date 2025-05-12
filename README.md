# EEG Signal Processing and Classification Pipeline

This project implements a complete pipeline for processing EEG signals and training a Vision Transformer (ViT) model for classification. The pipeline consists of two main components:

1. Signal Processing and Spectrogram Generation (`generate_graph.py`)
2. Model Training with Hybrid Loss (`TT_Hybrid_Loss_Cha.py`)

## Dataset

This project uses the [PhysioNet Challenge 2018](https://physionet.org/content/challenge-2018/1.0.0/) dataset, which focuses on automatic detection of sleep arousals. The dataset includes:
- Polysomnographic recordings from sleep studies
- Multiple physiological signals including EEG, ECG, and EMG
- Expert-annotated arousal events
- Training and test sets with corresponding labels

The dataset is particularly valuable for sleep research as it contains:
- High-quality physiological signals
- Expert-annotated arousal events
- Multiple recording channels
- Standardized scoring criteria

## Model Architecture

This project uses the [ViT-Hybrid-Base-Bit-384](https://huggingface.co/google/vit-hybrid-base-bit-384) model from Google, which is a hybrid Vision Transformer architecture that combines:
- A convolutional backbone (BiT)
- Transformer encoder layers
- Pre-trained on ImageNet-21k and fine-tuned on ImageNet-1k

The model is particularly well-suited for our EEG spectrogram classification task due to its:
- Hybrid architecture that can capture both local and global features
- Pre-trained weights that provide strong feature extraction capabilities
- 384x384 input resolution that matches our spectrogram dimensions

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
EEG_PATH = Path("/path/to/eeg/data")  # Path to PhysioNet Challenge 2018 data
OUTPUT_PATH = Path.cwd() / "Graph_Test"  # Output directory
PROCESSES = 10  # Number of parallel processes
MODE = 'cwt'  # Options: 'cwt', 'ssq', 'both'
USE_MNE = False  # Whether to apply MNE ICA denoising
CHANNELS = None  # List channels or None for all
SAMPLE_SEC = 30  # Segment length in seconds
FREQ_RANGE = (0.1, 35)  # Frequency range for spectrograms
IMAGE_SIZE = 384  # Output image size (matches ViT-Hybrid input size)
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
MODEL_ID = "google/vit-hybrid-base-bit-384"  # Base model from Hugging Face
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
- Vision Transformer (ViT) with hybrid architecture from [Hugging Face](https://huggingface.co/google/vit-hybrid-base-bit-384)
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
- The ViT-Hybrid model is pre-trained on ImageNet and fine-tuned for our specific EEG classification task
- The PhysioNet Challenge 2017 dataset provides a standardized benchmark for sleep arousal detection

## License

[Specify your license here]

## Citation

If you use this code in your research, please cite:

### Dataset Citation
```bibtex
@article{ghassemi2018snooze,
  title={You snooze, you win: the physionet/computing in cardiology challenge 2018},
  author={Ghassemi, Mohammad M and Moody, Benjamin E and Lehman, Li-wei H and Song, Chengyu and Li, Qiao and Sun, Haoqi and Westover, M Brandon and Clifford, Gari D},
  journal={2018 Computing in Cardiology Conference (CinC)},
  volume={45},
  pages={1--4},
  year={2018},
  publisher={IEEE},
  doi={10.22489/CinC.2018.049}
}
```

### Model Citation
```bibtex
@misc{wu2020visual,
      title={Visual Transformers: Token-based Image Representation and Processing for Computer Vision}, 
      author={Bichen Wu and Chenfeng Xu and Xiaoliang Dai and Alvin Wan and Peizhao Zhang and Zhicheng Yan and Masayoshi Tomizuka and Joseph Gonzalez and Kurt Keutzer and Peter Vajda},
      year={2020},
      eprint={2006.03677},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
``` 
