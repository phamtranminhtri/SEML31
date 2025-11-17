import librosa
import numpy as np
from scipy.ndimage import median_filter as scipy_median_filter
from typing import Tuple
try:
    from preprocessing import SAMPLE_RATE, HOP_LENGTH
except ImportError:
    print("Warning: Could not import constants. Using fallback default values.")
    SAMPLE_RATE = 22050
    HOP_LENGTH = 512 

def extract_features(audio_path: str, sr: int = SAMPLE_RATE, hop_length: int = HOP_LENGTH) -> np.ndarray:
    
    y, sr = librosa.load(audio_path, sr=sr)
    y_harm, _ = librosa.effects.hpss(y, margin=2.0)

    chroma = librosa.feature.chroma_cqt(
        y=y_harm, sr=sr,
        hop_length=hop_length,
        bins_per_octave=48
    ).T

    tonnetz = librosa.feature.tonnetz(
        y=y_harm, sr=sr,
        hop_length=hop_length
    ).T

    spectral_contrast = librosa.feature.spectral_contrast(
        y=y_harm, sr=sr,
        hop_length=hop_length,
        n_bands=6
    ).T

    chroma = scipy_median_filter(chroma, size=(3, 1))
    tonnetz = scipy_median_filter(tonnetz, size=(3, 1))
    spectral_contrast = scipy_median_filter(spectral_contrast, size=(3, 1))

    chroma = librosa.util.normalize(chroma, norm=2, axis=1)
    tonnetz = librosa.util.normalize(tonnetz, norm=2, axis=1)
    spectral_contrast = librosa.util.normalize(spectral_contrast, norm=2, axis=1)

    base = np.hstack([chroma, tonnetz, spectral_contrast])
    
    delta = librosa.feature.delta(base.T)
    delta2 = librosa.feature.delta(base.T, order=2)

    delta = delta.T
    delta2 = delta2.T

    features = np.hstack([base, delta, delta2])
    return features