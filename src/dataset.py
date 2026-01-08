import os
from typing import List, Tuple, Optional

import numpy as np
import librosa


AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")


def list_audio_files(root_dir: str, max_files: Optional[int] = None) -> List[str]:
    paths = []
    for r, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(AUDIO_EXTS):
                paths.append(os.path.join(r, f))
    paths.sort()
    if max_files is not None:
        paths = paths[:max_files]
    return paths


def extract_mfcc_features(
    path: str,
    sr: int = 22050,
    duration: float = 30.0,
    n_mfcc: int = 40,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Returns a fixed-length vector: [mfcc_mean (n_mfcc), mfcc_std (n_mfcc)] => 2*n_mfcc
    """
    y, _ = librosa.load(path, sr=sr, mono=True, duration=duration)
    if y.size == 0:
        raise ValueError("Empty audio")

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)

    feat = np.concatenate([mfcc_mean, mfcc_std], axis=0).astype(np.float32)
    return feat


def load_audio_feature_matrix(
    audio_dir: str,
    max_files: Optional[int] = 500,
    sr: int = 22050,
    duration: float = 30.0,
    n_mfcc: int = 20,
) -> Tuple[np.ndarray, List[str]]:
    """
    Loads MFCC mean/std features for all audio files under audio_dir.
    Returns:
      X: (N, 2*n_mfcc)
      ids: list of file paths (or IDs) aligned with X
    """
    files = list_audio_files(audio_dir, max_files=max_files)
    X = []
    ids = []

    for p in files:
        try:
            feat = extract_mfcc_features(p, sr=sr, duration=duration, n_mfcc=n_mfcc)
            X.append(feat)
            ids.append(p)
        except Exception:
            # Skip unreadable/broken files to keep pipeline robust
            continue

    if len(X) == 0:
        raise RuntimeError(
            f"No audio features extracted. Check audio_dir='{audio_dir}' and file formats."
        )

    X = np.stack(X, axis=0)
    return X, ids



#Medium task
import numpy as np
import librosa


def extract_logmel_spectrogram(
    path: str,
    sr: int = 22050,
    duration: float = 30.0,
    n_mels: int = 64,
    hop_length: int = 512,
    n_fft: int = 2048,
    max_frames: int = 256,
) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True, duration=duration)
    if y.size == 0:
        raise ValueError("Empty audio")

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft, power=2.0)
    logmel = librosa.power_to_db(mel, ref=np.max)  # (n_mels, T)

    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-8)

    T = logmel.shape[1]
    if T >= max_frames:
        logmel = logmel[:, :max_frames]
    else:
        logmel = np.pad(logmel, ((0, 0), (0, max_frames - T)), mode="constant")

    return logmel.astype(np.float32)[..., None]  # (n_mels, max_frames, 1)


def load_spectrogram_tensor(
    audio_dir: str,
    max_files: int = 2000,
    sr: int = 22050,
    duration: float = 30.0,
    n_mels: int = 64,
    max_frames: int = 256,
):
    files = list_audio_files(audio_dir, max_files=max_files)
    X, ids = [], []
    skipped = 0
    for p in files:
        try:
            X.append(extract_logmel_spectrogram(p, sr=sr, duration=duration, n_mels=n_mels, max_frames=max_frames))
            ids.append(p)
        except Exception:
            skipped += 1
            continue
    if len(X) == 0:
        raise RuntimeError(f"No spectrograms extracted. Check audio_dir={audio_dir}")
    X = np.stack(X, axis=0)
    print(f"[load_spectrogram_tensor] attempted={len(files)} loaded={len(X)} skipped={skipped}")
    return X, ids


def load_lyrics_map(lyrics_dir: str):
    """
    Expects .txt lyrics with filename stem matching audio filename stem:
      data/lyrics/blues.00000.txt  matches audio blues.00000.wav
    """
    lyr = {}
    if not lyrics_dir or not os.path.exists(lyrics_dir):
        return lyr

    for r, _, files in os.walk(lyrics_dir):
        for f in files:
            if f.lower().endswith(".txt"):
                stem = os.path.splitext(f)[0]
                p = os.path.join(r, f)
                try:
                    with open(p, "r", encoding="utf-8") as fh:
                        lyr[stem] = fh.read()
                except Exception:
                    continue
    return lyr


def align_lyrics_to_audio_ids(audio_ids, lyrics_map):
    texts = []
    for p in audio_ids:
        stem = os.path.splitext(os.path.basename(p))[0]
        texts.append(lyrics_map.get(stem, ""))
    return texts

