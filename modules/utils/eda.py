import os
import glob
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def summarize_dataset(dataset_dir: str) -> Tuple[List[str], List[str]]:
    """
    Scan dataset directory, print summary and return list of .lab and .mp3 files.
    """
    if not os.path.exists(dataset_dir):
        print(f"Error: Directory not found at {dataset_dir}")
        return [], []

    lab_files = sorted(glob.glob(os.path.join(dataset_dir, '*.lab')))
    audio_files = sorted(glob.glob(os.path.join(dataset_dir, '*.mp3')))

    print(f"--- Dataset Directory Analysis ---")
    print(f"Path: {dataset_dir}")
    print(f"Number of '.lab' files: {len(lab_files)}")
    print(f"Number of audio files: {len(audio_files)}")
    
    if len(lab_files) != len(audio_files):
        print(f"Warning: Number of lab and audio files don't match!")

    return lab_files, audio_files

def load_and_preview_lab(lab_file_path: str, num_rows: int = 10) -> pd.DataFrame:
    """
    Load a .lab file and display the first rows.
    """
    print(f"--- File Preview: {os.path.basename(lab_file_path)} ---")
    try:
        df_lab = pd.read_csv(
            lab_file_path,
            sep=r'\s+',
            header=None,
            names=['start_time', 'end_time', 'chord_label']
        )
        print(f"Total chord segments: {len(df_lab)}")
        display(df_lab.head(num_rows))
        return df_lab
    except Exception as e:
        print(f"Error reading file {lab_file_path}: {e}")
        return pd.DataFrame()

def calculate_global_chord_counts(lab_files: List[str]) -> pd.Series:
    """
    Read all .lab files and count frequency of each chord.
    """
    all_chords = []
    print("Aggregating all chords from .lab files...")
    for lab_file in tqdm(lab_files, desc="Reading Lab files"):
        try:
            df_lab = pd.read_csv(
                lab_file,
                sep=r'\s+',
                header=None,
                usecols=[2],
                names=['chord_label']
            )
            all_chords.extend(df_lab['chord_label'].tolist())
        except Exception:
            continue
            
    return pd.Series(all_chords).value_counts()

def plot_chord_distribution(chord_counts: pd.Series, top_n: int = 20) -> plt.Figure:
    """
    Plot Top N most common chords.
    """
    top_chords = chord_counts.head(top_n)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        x=top_chords.index,
        y=top_chords.values,
        palette="flare",
        hue=top_chords.index,
        legend=False,
        ax=ax
    )
    ax.set_title(f'Top {top_n} Most Common Chords', fontsize=15)
    ax.set_xlabel('Chord', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.tick_params(axis='x', rotation=45, ha='right')
    
    plt.tight_layout()
    return fig

def analyze_and_plot_durations(audio_files: List[str]) -> plt.Figure:
    """
    Analyze and plot song duration distribution.
    """
    print(f"--- Song Duration Analysis ---")
    durations = []
    for audio_file in tqdm(audio_files, desc="Analyzing durations"):
        if os.path.exists(audio_file):
            try:
                duration = librosa.get_duration(filename=audio_file)
                durations.append(duration)
            except Exception as e:
                print(f"Error loading {audio_file}: {e}")

    durations_np = np.array(durations)
    
    print(f"Number of songs analyzed: {len(durations_np)}")
    print(f"Average duration: {durations_np.mean():.2f} seconds")
    print(f"Shortest duration: {durations_np.min():.2f} seconds")
    print(f"Longest duration: {durations_np.max():.2f} seconds")
    print(f"Standard deviation: {durations_np.std():.2f} seconds")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(durations_np, bins=20, kde=False, color='#6932a8', ax=ax)
    ax.set_xlabel('Duration (seconds)', fontsize=12)
    ax.set_ylabel('Number of songs', fontsize=12)
    ax.set_title('Song Duration Distribution', fontsize=15)
    ax.axvline(durations_np.mean(), color='red', linestyle='--', label=f'Average: {durations_np.mean():.1f}s')
    ax.legend()
    
    plt.tight_layout()
    return fig

def plot_audio_visualizations(audio_file: str, sr: int = 22050) -> plt.Figure:
    """
    Plot 3 visualizations: Waveform, Spectrogram, and Chromagram for an audio file.
    """
    print(f"--- File Visualization: {os.path.basename(audio_file)} ---")
    try:
        y, sr = librosa.load(audio_file, sr=sr)
    except Exception as e:
        print(f"Error loading {audio_file}: {e}")
        return plt.Figure()

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    librosa.display.waveshow(y, sr=sr, ax=axes[0], color='#009999', x_axis='time')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title(f'Waveform: {os.path.basename(audio_file)}')

    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img_spec = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=axes[1], cmap='viridis')
    axes[1].set_title('Spectrogram')
    fig.colorbar(img_spec, ax=axes[1], format='%+2.0f dB')

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
    img_chroma = librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma', ax=axes[2], cmap='coolwarm')
    axes[2].set_title('Chromagram')
    fig.colorbar(img_chroma, ax=axes[2])

    plt.tight_layout()
    return fig

def analyze_and_plot_chord_positions(lab_files: List[str]) -> plt.Figure:
    """
    Analyze and plot chords by 3 positions: Start, Middle, End of song.
    """
    print(f"--- Chord Position Analysis ---")
    chord_positions: Dict[str, List[str]] = {'start': [], 'middle': [], 'end': []}

    for lab_file in tqdm(lab_files, desc="Analyzing positions"):
        try:
            df_lab = pd.read_csv(lab_file, sep=r'\s+', header=None,
                                 names=['start_time', 'end_time', 'chord_label'])
            total_duration = df_lab['end_time'].max()
            if total_duration == 0:
                continue

            for _, row in df_lab.iterrows():
                position_ratio = row['start_time'] / total_duration
                if position_ratio < 0.33:
                    chord_positions['start'].append(row['chord_label'])
                elif position_ratio < 0.67:
                    chord_positions['middle'].append(row['chord_label'])
                else:
                    chord_positions['end'].append(row['chord_label'])
        except Exception:
            continue

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    positions = ['start', 'middle', 'end']
    titles = ['Start (0-33%)', 'Middle (33-67%)', 'End (67-100%)']

    for ax, pos, title in zip(axes, positions, titles):
        top_chords = pd.Series(chord_positions[pos]).value_counts().head(10)
        sns.barplot(x=top_chords.index, y=top_chords.values,
                    palette='crest', hue=top_chords.index, legend=False, ax=ax)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel('Chord')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    return fig

def analyze_and_plot_transitions(lab_files: List[str], top_n: int = 20) -> plt.Figure:
    """
    Analyze and plot Top N most common chord transitions.
    """
    print(f"--- Chord Transition Analysis (Top {top_n}) ---")
    transitions: Dict[str, int] = {}

    for lab_file in tqdm(lab_files, desc="Analyzing transitions"):
        try:
            df_lab = pd.read_csv(lab_file, sep=r'\s+', header=None,
                                 names=['start_time', 'end_time', 'chord_label'])

            for i in range(len(df_lab) - 1):
                chord_from = df_lab.iloc[i]['chord_label']
                chord_to = df_lab.iloc[i+1]['chord_label']
                
                if chord_from == chord_to:
                    continue

                key = f"{chord_from} → {chord_to}"
                transitions[key] = transitions.get(key, 0) + 1
        except Exception:
            continue

    top_transitions = pd.Series(transitions).nlargest(top_n)

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(x=top_transitions.values, y=top_transitions.index,
                palette='viridis', hue=top_transitions.index, legend=False, ax=ax)
    ax.set_title(f'Top {top_n} Chord Transitions (Excluding Self-loops)', fontsize=15)
    ax.set_xlabel('Count')
    ax.set_ylabel('Transition')
    
    plt.tight_layout()
    return fig