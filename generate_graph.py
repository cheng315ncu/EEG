"""
process_eeg.py

Automatically process EEG WFDB records:
- Optional artifact removal via MNE-ICA
- Generate CWT and/or SSQ spectrograms per channel
- Save outputs under ./Graph/<record>/<channel>/{cwt,ssq}/
- Save labels per record in Graph/<record>/labels.txt

Configuration is embedded; adjust EEG_PATH as needed.
"""
import gc
import logging
import multiprocessing
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom
from scipy.signal import butter, filtfilt
import wfdb
import mne
from ssqueezepy import cwt, ssq_cwt
from ssqueezepy.experimental import scale_to_freq
from PIL import Image
from tqdm import tqdm

# --- Configuration ---
EEG_PATH = Path(
    "/home/phys/data/PHYS/EGG data process/challenge-2018-1.0.0.physionet.org/training"
)
# Base output path: current working directory / Graph
OUTPUT_PATH = Path.cwd() / "Graph_Test"
PROCESSES = 10
WAVELET = 'morlet'
MODE = 'cwt'  # options: 'cwt', 'ssq', 'both'
USE_MNE = False  # whether to apply MNE ICA denoising
CHANNELS = None  # list channels or None for all
SAMPLE_SEC = 30
FREQ_RANGE = (0.1, 35)
IMAGE_SIZE = 384

# MNE channel type mapping
CHANNEL_TYPES = {
    'SaO2': 'misc', 'ABD': 'emg', 'CHEST': 'emg', 'Chin1-Chin2': 'emg',
    'AIRFLOW': 'misc', 'ECG': 'ecg', 'E1-M2': 'eog',
    'C4-M1': 'eeg', 'C3-M2': 'eeg', 'F3-M2': 'eeg', 'F4-M1': 'eeg', "O1-M2": "eeg", "O2-M1": "eeg"
}

# Initialize logging
logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO
)

def release_memory():
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass


def get_record_name_list():
    return sorted([p.name for p in EEG_PATH.iterdir() if p.is_dir()])


def read_eeg(record_name):
    return wfdb.rdrecord(str(EEG_PATH / record_name / record_name))


def read_arousal(record_name):
    return wfdb.rdann(str(EEG_PATH / record_name / record_name), 'arousal')


def get_valid_informations(notes):
    labels, indices = [], []
    nesting = 0
    for idx, note in enumerate(notes):
        if isinstance(note, str):
            if note.startswith('('): nesting += 1; continue
            if note.endswith(')'): nesting = max(nesting-1, 0); continue
            if nesting == 0:
                labels.append(note); indices.append(idx)
    return labels, indices


def band_filter(data, lowcut, highcut, fs, order=4, btype='band'):
    nyq = 0.5 * fs
    freqs = [f/nyq for f in (lowcut, highcut) if f is not None]
    b, a = butter(order, freqs, btype=btype)
    return filtfilt(b, a, data)


def compute_image(array):
    y, x = array.shape
    return zoom(array, (IMAGE_SIZE / y, IMAGE_SIZE / x), order=1)


def save_spectrogram(img_array, out_file):
    import matplotlib.pyplot as plt
    plt.switch_backend('Agg')
    fig = plt.figure(figsize=(IMAGE_SIZE/100, IMAGE_SIZE/100), dpi=100)
    plt.imshow(img_array, aspect='auto', cmap='jet')
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(str(out_file), dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    im = Image.open(str(out_file))
    if im.size != (IMAGE_SIZE, IMAGE_SIZE):
        im.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS).save(str(out_file))


def save_cwt_and_ssq(data, fs, scales, record_dir, channel, segment_idx):
    # preprocess
    sig = data.astype(np.float32) / 1e6
    sig = band_filter(sig, 45, 55, fs, btype='bandstop')
    sig = band_filter(sig, 0.3, 35, fs, btype='band')
    sig -= sig.mean(); sig /= np.max(np.abs(sig))
    
    # CWT
    if MODE in ('cwt', 'both'):
        wx, sc = cwt(sig, WAVELET, scales=scales, fs=fs)
        freqs = scale_to_freq(sc, WAVELET, fs=fs, N=len(sig))
        mask = (freqs >= FREQ_RANGE[0]) & (freqs <= FREQ_RANGE[1])
        arr = np.abs(wx[mask]); img = compute_image(arr)
        cwt_dir = record_dir / channel / 'cwt'
        cwt_dir.mkdir(parents=True, exist_ok=True)
        save_spectrogram(img, cwt_dir / f"{segment_idx}.jpg")

    # SSQ
    if MODE in ('ssq', 'both'):
        Tx, _, ssq_freqs, sc2 = ssq_cwt(sig, WAVELET, scales=scales, fs=fs)
        mask2 = (ssq_freqs >= FREQ_RANGE[0]) & (ssq_freqs <= FREQ_RANGE[1])
        arr2 = np.abs(Tx[mask2]); img2 = compute_image(arr2)
        ssq_dir = record_dir / channel / 'ssq'
        ssq_dir.mkdir(parents=True, exist_ok=True)
        save_spectrogram(img2, ssq_dir / f"{segment_idx}.jpg")


def remove_artifacts(p_signal, fs, ch_names):
    data = p_signal.T
    ch_types = [CHANNEL_TYPES.get(ch, 'misc') for ch in ch_names]
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=ch_types)
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.filter(0.3, 35., fir_design='firwin', verbose=False)
    ica = mne.preprocessing.ICA(n_components=0.95, random_state=97, max_iter='auto', verbose=False)
    ica.fit(raw)
    for kind, func in [('EOG', ica.find_bads_eog), ('ECG', ica.find_bads_ecg)]:
        if kind in ch_names:
            idxs, _ = func(raw, ch_name=kind, method='correlation', threshold='auto', verbose=False)
            ica.exclude.extend(idxs)
    cleaned = ica.apply(raw.copy(), verbose=False).get_data().T
    return cleaned


def process_record(j):
    records = get_record_name_list(); record = records[j]
    logging.info(f"Starting {record}")
    try:
        rec = read_eeg(record); ann = read_arousal(record)
        fs = rec.fs; p_signal = rec.p_signal #; ch_names = rec.sig_name
        ch_names = ["C4-M1"]
        labels, nums = get_valid_informations(ann.aux_note)

        record_dir = OUTPUT_PATH / record
        record_dir.mkdir(parents=True, exist_ok=True)
        # save labels
        with open(record_dir / 'labels.txt', 'w') as f:
            for lab in labels: f.write(f"{lab}\n")

        targets = CHANNELS if CHANNELS else ch_names
        cleaned = remove_artifacts(p_signal, fs, ch_names) if USE_MNE else p_signal
        for ch in targets:
            if ch not in ch_names: continue
            idx = ch_names.index(ch)
            for i, n in enumerate(nums, 1):
                start = ann.sample[n]
                seg = cleaned[start:start + int(SAMPLE_SEC * fs), idx]
                save_cwt_and_ssq(seg, fs, 'log-piecewise', record_dir, ch, i)
        return f"Processed {record}"
    except Exception as e:
        logging.error(f"Error processing {record}: {e}")
        release_memory(); return f"Failed {record}: {e}"

if __name__ == '__main__':
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    records = get_record_name_list()
    with multiprocessing.Pool(PROCESSES) as pool:
        for res in tqdm(
            pool.imap_unordered(process_record, range(len(records))),
            total=len(records), desc='Overall', ncols=80
        ):
            logging.info(res)
