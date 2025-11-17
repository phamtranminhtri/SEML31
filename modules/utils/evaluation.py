import numpy as np
from scipy.signal import medfilt
from tqdm import tqdm
import sys
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

try:
    from preprocessing import CHORD_TO_ID, ID_TO_CHORD, simplify_chord
except ImportError:
    print("Warning: Could not import preprocessing constants.")
    CHORD_TO_ID = {}
    ID_TO_CHORD = {}
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
    """
    Tính toán và in ra độ chính xác frame-by-frame.
    """
    pred_flat = []
    true_flat = []
    
    for key in pred_dict:
        if key in true_dict:
            pred_flat.extend(pred_dict[key])
            true_flat.extend(true_dict[key])
            
    if not true_flat:
        print("Error: No matching labels found to calculate accuracy.")
        return
        
    acc = accuracy_score(true_flat, pred_flat)
    
    print(f"\n--- 📊 Evaluation Metrics ---")
    print(f"Overall Frame-by-Frame Accuracy: {acc * 100:.2f}%")
    
    labels = sorted(list(set(true_flat)))
    
    print("\nClassification Report (Sample):")
    # Chỉ in report cho các nhãn chính, vì 25 nhãn là quá dài
    try:
        sample_labels = sorted(list(set(labels) & set(['N', 'C:maj', 'G:maj', 'A:min', 'F:maj'])))
        if not sample_labels:
             sample_labels = labels[:5] # Fallback
    
        print(classification_report(true_flat, pred_flat, labels=sample_labels, zero_division=0))
    except Exception as e:
        print(f"Could not generate sample classification report: {e}")


def print_chord_comparison(
    predicted_dict: dict,
    true_dict: dict,
    chords_per_line: int = 10,
    output_file=sys.stdout
):
    """
    In ra so sánh trực quan, đã căn chỉnh.
    """
    max_len = 0
    all_chord_lists = list(predicted_dict.values()) + list(true_dict.values())
    for song_list in all_chord_lists:
        for chord in song_list:
            max_len = max(max_len, len(chord))

    cell_width = max_len + 1

    for song_key in predicted_dict.keys():
        if song_key not in true_dict:
            continue

        pred_chords = predicted_dict[song_key]
        true_chords = true_dict[song_key]
        num_frames = len(true_chords)

        print(f"\n" + "=" * 80, file=output_file)
        print(f"📊 SONG COMPARISON: {song_key}", file=output_file)
        print("=" * 80, file=output_file)

        for i in range(0, num_frames, chords_per_line):
            start = i
            end = min(i + chords_per_line, num_frames)

            true_chunk = true_chords[start:end]
            pred_chunk = pred_chords[start:end]

            true_line = "".join([f"{chord:<{cell_width}}" for chord in true_chunk])
            pred_line = "".join([f"{chord:<{cell_width}}" for chord in pred_chunk])

            print(f"  [Frames {start:04d}-{end - 1:04d}]", file=output_file)
            print(f"  ✅ True: | {true_line}", file=output_file)
            print(f"  🎯 Pred: | {pred_line}", file=output_file)
            print(file=output_file)