# import os
# import numpy as np
# import h5py

# def _safe_song_scalar(h5f, key: str, default=np.nan) -> float:
#     try:
#         return float(h5f["analysis"]["songs"][key][0])
#     except Exception:
#         return float(default)

# def load_msd_h5_features(audio_dir: str, max_files: int | None = None):
#     """
#     Reads MSD .h5 files from data/audio/ and returns:
#       X: (N, D) float32
#       ids: list of track IDs (file stems)
#       feature_names: list[str]
#     Uses stable SONG-LEVEL features for Easy Task.
#     """
#     files = [f for f in os.listdir(audio_dir) if f.lower().endswith(".h5")]
#     files.sort()
#     if max_files is not None:
#         files = files[:max_files]

#     feature_names = ["tempo", "loudness", "duration", "key", "mode", "time_signature"]

#     X = []
#     ids = []

#     for fname in files:
#         path = os.path.join(audio_dir, fname)
#         try:
#             with h5py.File(path, "r") as h5f:
#                 tempo = _safe_song_scalar(h5f, "tempo")
#                 loudness = _safe_song_scalar(h5f, "loudness")
#                 duration = _safe_song_scalar(h5f, "duration")
#                 key = _safe_song_scalar(h5f, "key")
#                 mode = _safe_song_scalar(h5f, "mode")
#                 time_signature = _safe_song_scalar(h5f, "time_signature")

#             vec = [tempo, loudness, duration, key, mode, time_signature]

#             # skip samples with too many missing values
#             if np.isnan(vec).sum() >= 3:
#                 continue

#             X.append(vec)
#             ids.append(fname.replace(".h5", ""))

#         except Exception:
#             # skip corrupted/unexpected files
#             continue

#     X = np.asarray(X, dtype=np.float32)
#     return X, ids, feature_names



import os
import numpy as np
import h5py

def _safe_song_scalar(h5f, key: str, default=np.nan) -> float:
    try:
        return float(h5f["analysis"]["songs"][key][0])
    except Exception:
        return float(default)

def load_msd_h5_features(audio_dir: str, max_files: int | None = None):
    """
    Recursively loads MSD .h5 files from data/audio/ (including nested folders).
    Returns:
      X: (N, D) float32
      ids: list[str]
      feature_names: list[str]
    """

    feature_names = ["tempo", "loudness", "duration", "key", "mode", "time_signature"]

    # 1) Find all .h5 files recursively
    h5_paths = []
    for root, _, files in os.walk(audio_dir):
        for fname in files:
            if fname.lower().endswith(".h5"):
                h5_paths.append(os.path.join(root, fname))

    h5_paths.sort()
    if max_files is not None:
        h5_paths = h5_paths[:max_files]

    X, ids = [], []

    # 2) Extract features from each .h5
    for path in h5_paths:
        try:
            with h5py.File(path, "r") as h5f:
                tempo = _safe_song_scalar(h5f, "tempo")
                loudness = _safe_song_scalar(h5f, "loudness")
                duration = _safe_song_scalar(h5f, "duration")
                key = _safe_song_scalar(h5f, "key")
                mode = _safe_song_scalar(h5f, "mode")
                time_signature = _safe_song_scalar(h5f, "time_signature")

            vec = [tempo, loudness, duration, key, mode, time_signature]

            # Skip samples with too many missing values
            if np.isnan(vec).sum() >= 3:
                continue

            X.append(vec)
            ids.append(os.path.basename(path).replace(".h5", ""))

        except Exception:
            continue

    X = np.asarray(X, dtype=np.float32)
    return X, ids, feature_names
