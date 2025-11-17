import numpy as np
from scipy.signal import medfilt
from tqdm import tqdm
import sys
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

try:
    from preprocessing import CHORD_TO_ID, ID_TO_CHORD, simplify_chord, CHORD_STATES
except ImportError:
    print("Warning: Could not import preprocessing constants.")
    CHORD_TO_ID = {'N': 0}
    ID_TO_CHORD = {0: 'N'}
    CHORD_STATES = ['N']
    def simplify_chord(c): return c


def compute_log_likelihoods(GMM_Models, features):
    n_states = len(GMM_Models)
    n_frames = features.shape[0]
    log_B = np.full((n_states, n_frames), -np.inf)
    for i in range(n_states):
        gmm = GMM_Models[i]
        if gmm is not None:
            log_B[i, :] = gmm.score_samples(features)
    return log_B

def viterbi_log(pi, A, log_B):
    n_states = A.shape[0]
    T = log_B.shape[1]
    delta = np.zeros((T, n_states))
    phi = np.zeros((T, n_states), dtype=int)
    log_pi = np.log(pi + 1e-10)
    log_A = np.log(A + 1e-10)
    delta[0, :] = log_pi + log_B[:, 0]
    for t in range(1, T):
        for j in range(n_states):
            temp = delta[t-1, :] + log_A[:, j]
            delta[t, j] = np.max(temp) + log_B[j, t]
            phi[t, j] = np.argmax(temp)
    q_star = np.zeros(T, dtype=int)
    q_star[T-1] = np.argmax(delta[T-1, :])
    for t in range(T-2, -1, -1):
        q_star[t] = phi[t+1, q_star[t+1]]
    return q_star


def predict_on_test_set(X_test_list, y_test_list, pi, A, GMM_Models):
    all_predicted_chords = {}
    all_true_labels = {}
    desc = "Predicting on Test Set"
    for i, (features, true_labels) in tqdm(
            enumerate(zip(X_test_list, y_test_list)),
            total=len(X_test_list), desc=desc):
        
        log_B = compute_log_likelihoods(GMM_Models, features)
        expected_shape = (len(GMM_Models), features.shape[0])
        if log_B.shape != expected_shape:
            print(f"Warning: log_B shape mismatch {log_B.shape}. Skipping song {i}.")
            continue

        predicted_state_ids = viterbi_log(pi, A, log_B)
        predicted_state_ids = medfilt(predicted_state_ids, kernel_size=5).astype(int)

        song_key = f'song_{i}'
        all_predicted_chords[song_key] = [ID_TO_CHORD.get(sid, 'N') for sid in predicted_state_ids]
        
        true_ids = []
        for ch in true_labels:
            simplified_ch = simplify_chord(ch)
            true_ids.append(CHORD_TO_ID.get(simplified_ch, CHORD_TO_ID['N']))
        all_true_labels[song_key] = [ID_TO_CHORD.get(tid, 'N') for tid in true_ids]
        
    return all_predicted_chords, all_true_labels


def calculate_accuracy(pred_dict, true_dict):
    pred_flat = []
    true_flat = []
    for key in pred_dict:
        if key in true_dict:
            pred_flat.extend(pred_dict[key])
            true_flat.extend(true_dict[key])
            
    if not true_flat:
        print("Error: No matching labels to calculate accuracy.")
        return 0.0, [], []
        
    acc = accuracy_score(true_flat, pred_flat)
    
    print(f"\n--- Evaluation Metrics ---")
    print(f"Overall Accuracy: {acc * 100:.2f}%")
    
    return acc, true_flat, pred_flat

def plot_confusion_matrix(y_true_flat, y_pred_flat, class_names_list):
    print(f"\n--- Confusion Matrix ---")
    
    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=class_names_list)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=False,
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names_list,
        yticklabels=class_names_list,
        ax=ax
    )
    
    ax.set_title(f'Confusion Matrix', fontsize=16)
    ax.set_ylabel('True Label', fontsize=14)
    ax.set_xlabel('Predicted Label', fontsize=14)
    ax.tick_params(axis='x', rotation=90, labelsize=8)
    ax.tick_params(axis='y', rotation=0, labelsize=8)
    
    plt.tight_layout()
    return fig