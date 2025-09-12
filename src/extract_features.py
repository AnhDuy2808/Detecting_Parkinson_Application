import pandas as pd
import numpy as np
import parselmouth
from parselmouth.praat import call
import librosa
import nolds
import os
from tqdm import tqdm
from scipy.stats import entropy
from scipy.io.wavfile import read
import time


def compute_dfa(signal):
    """Compute DFA (Detrended Fluctuation Analysis)"""
    return nolds.dfa(signal) if len(signal) > 10 else np.nan


def compute_d2(signal):
    """Compute correlation dimension D2"""
    return nolds.corr_dim(signal, emb_dim=10) if len(signal) > 10 else np.nan


def compute_nonlinear_measures(f0_series):
    """Compute spread1, spread2, and PPE"""
    if len(f0_series) > 10:
        spread1 = np.mean(f0_series) - np.std(f0_series)
        spread2 = np.log(np.std(f0_series) + 1e-6)  # Avoid log(0)
        hist, _ = np.histogram(f0_series, bins=10, density=True)
        PPE = entropy(hist)  # Probability density entropy
        return spread1, spread2, PPE
    return np.nan, np.nan, np.nan  # Default values


def estimate_f0_range(audio_path):
    """Estimate the minimum and maximum fundamental frequency (f0min, f0max) from an audio file."""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        sound = parselmouth.Sound(audio_path)
        pitch = sound.to_pitch_ac(time_step=0.01, very_accurate=True)
        f0_values = pitch.selected_array['frequency']
        f0_values = f0_values[f0_values > 0]  # Remove zero values
        if len(f0_values) == 0:
            print(f"No valid pitch values found in {audio_path}")
            return None, None
        f0min = np.min(f0_values)
        f0max = np.max(f0_values)
        return f0min, f0max
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None, None


def extract_features(file_path):
    features = {}
    try:
        start = time.perf_counter()
        y, sr = librosa.load(file_path, sr=16000)
        if len(y) / sr < 0.5:  # Check for minimum audio length
            print(f"   Audio too short: {len(y)/sr:.2f}s")
            raise ValueError("Audio file is too short")
        if np.mean(np.abs(y)) < 1e-6:  # Check for near-silent audio
            print(f"   Audio is near-silent: mean amplitude {np.mean(np.abs(y)):.6f}")
            raise ValueError("Audio file is near-silent")
        
        sound = parselmouth.Sound(file_path)
        # print(f"   Audio length: {len(y)/sr:.2f}s, Sample rate: {sr} Hz")

        # Fundamental frequency (F0)
        t0 = time.perf_counter()
        f0min, f0max = estimate_f0_range(file_path)
        if f0min is None or f0max is None:
            print(f"   Failed to estimate f0 range for {file_path}")
            raise ValueError("Invalid f0 range")

        pitch = call(sound, "To Pitch", 0.0, f0min, f0max)
        f0_values = pitch.selected_array['frequency']
        f0_values = f0_values[f0_values > 0]  # Remove zero values
        if len(f0_values) == 0:
            print(f"   No valid pitch values for {file_path}")
            raise ValueError("No valid pitch values")

        features['MDVP:Fo(Hz)'] = np.mean(f0_values)
        features['MDVP:Fhi(Hz)'] = np.max(f0_values)
        features['MDVP:Flo(Hz)'] = np.min(f0_values)

        # Jitter & Shimmer
        t0 = time.perf_counter()
        pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
        features['MDVP:Jitter(%)'] = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        features['MDVP:Jitter(Abs)'] = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
        features['MDVP:RAP'] = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        features['MDVP:PPQ'] = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
        features['Jitter:DDP'] = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)

        features['MDVP:Shimmer'] = call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        features['MDVP:Shimmer(dB)'] = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        features['Shimmer:APQ3'] = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        features['Shimmer:APQ5'] = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        features['MDVP:APQ'] = call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        features['Shimmer:DDA'] = call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        # HNR & NHR
        t0 = time.perf_counter()
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.1, 1.0)
        features['HNR'] = call(harmonicity, "Get mean", 0, 0)
        features['NHR'] = (
            1 / features['HNR']
            if features['HNR'] > 0
            else 0
        )

        # DFA, D2, Nonlinear
        t0 = time.perf_counter()
        features['DFA'] = compute_dfa(f0_values)
        features['D2'] = compute_d2(f0_values)
        features['spread1'], features['spread2'], features['PPE'] = compute_nonlinear_measures(f0_values)

        # MFCC Features with correct naming
        t0 = time.perf_counter()
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            if i == 1:
                features['mean_MFCC_1st_coef'] = np.mean(mfcc[i])
            elif i == 2:
                features['mean_MFCC_2nd_coef'] = np.mean(mfcc[i])
            elif i == 3:
                features['mean_MFCC_3rd_coef'] = np.mean(mfcc[i])
            else:
                features[f'mean_MFCC_{i}th_coef'] = np.mean(mfcc[i])
        # print(f"   MFCC: {time.perf_counter() - t0:.2f}s")


    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        feature_names = [
            'locPctJitter', 'locAbsJitter', 'rapJitter', 'ppq5Jitter', 'ddpJitter',
            'locShimmer', 'locDbShimmer', 'apq3Shimmer', 'apq5Shimmer', 'apq11Shimmer', 'ddaShimmer',
            'meanHarmToNoiseHarmonicity', 'meanNoiseToHarmHarmonicity',
            'DFA', 'PPE',
            'mean_MFCC_0th_coef', 'mean_MFCC_1st_coef', 'mean_MFCC_2nd_coef', 'mean_MFCC_3rd_coef',
            'mean_MFCC_4th_coef', 'mean_MFCC_5th_coef', 'mean_MFCC_6th_coef', 'mean_MFCC_7th_coef',
            'mean_MFCC_8th_coef', 'mean_MFCC_9th_coef', 'mean_MFCC_10th_coef', 'mean_MFCC_11th_coef',
            'mean_MFCC_12th_coef',
            'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'spread1', 'spread2', 'D2'
        ]
        for fname in feature_names:
            features[fname] = np.nan
    return features


if __name__ == "__main__":
    # Read demographics file
    csv_file = "./data/segment_info.csv"
    df = pd.read_csv(csv_file)

    all_features = []
    print("🔍 Bắt đầu trích xuất đặc trưng...")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        file_path = os.path.join("segmented_audio", row["id"])
        if not os.path.exists(file_path):
            print(f"⚠️ Không tìm thấy file: {file_path}")
            continue

        feats = extract_features(file_path)

        # Add id, gender, and class
        feats['id'] = row['id']
        feats['class'] = row['label']

        all_features.append(feats)

    df_features = pd.DataFrame(all_features)
    df_features.to_csv("./data/extracted_features.csv", index=False)
    print("✅ Đã lưu đặc trưng vào extracted_features.csv")