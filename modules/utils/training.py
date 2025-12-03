import numpy as np
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

try:
    from preprocessing import CHORD_TO_ID, simplify_chord
except ImportError:
    print("Warning: Could not import preprocessing constants.")
    CHORD_TO_ID = {} 
    def simplify_chord(c): return c

def calc_hmm_parameters(label_list, n_states, epsilon=1e-8):
    pi = np.zeros(n_states)
    A = np.zeros((n_states, n_states))

    for label_seq in label_list:
        if not label_seq:
            continue

        first = CHORD_TO_ID[simplify_chord(label_seq[0])]
        pi[first] += 1

        for current, nxt in zip(label_seq[:-1], label_seq[1:]):
            cur_id = CHORD_TO_ID[simplify_chord(current)]
            nxt_id = CHORD_TO_ID[simplify_chord(nxt)]
            A[cur_id, nxt_id] += 1

    pi = pi + epsilon
    pi = pi / pi.sum()

    A_sum = A.sum(axis=1, keepdims=True)
    A_sum[A_sum == 0] = 1.0 # Tránh chia cho 0 nếu 1 trạng thái không bao giờ xuất hiện
    A = A + epsilon
    A = A / A.sum(axis=1, keepdims=True)

    return A, pi

def train_GMM(
    X_train_list, 
    y_train_list, 
    n_states, 
    n_components=3, 
    covariance_type='diag'
):
    state_data = [[] for _ in range(n_states)]

    for features, labels in zip(X_train_list, y_train_list):
        for frame_idx in range(len(labels)):
            state = labels[frame_idx]
            if isinstance(state, str):
                state = CHORD_TO_ID[simplify_chord(state)]
            
            feature_vector = features[frame_idx]
            state_data[state].append(feature_vector)

    emission_models = []
    
    for state_id in tqdm(range(n_states), desc="Training GMMs (Emission Model 'B')"):
        X_state = np.array(state_data[state_id])
        
        n_samples = X_state.shape[0]
        current_n_components = n_components

        if n_samples < n_components:
            print(f"Warning: State {state_id} has only {n_samples} samples. Reducing components.")
            current_n_components = max(1, n_samples) # Ít nhất là 1
        
        if n_samples == 0:
            print(f"Error: State {state_id} has 0 samples. Appending None.")
            emission_models.append(None)
            continue

        gmm = GaussianMixture(
            n_components=current_n_components,
            covariance_type=covariance_type,
            max_iter=100,
            random_state=72,
            n_init=3
        )
        gmm.fit(X_state)
        emission_models.append(gmm)

    return emission_models