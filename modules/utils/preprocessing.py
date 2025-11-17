import numpy as np
import re
from typing import List, Tuple

SAMPLE_RATE = 22050
HOP_LENGTH = 512

CHORD_ROOTS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
CHORD_TYPES = [':maj', ':min']
CHORD_STATES = [root + chord_type for root in CHORD_ROOTS for chord_type in CHORD_TYPES]
CHORD_STATES.append('N')

CHORD_TO_ID = {chord: i for i, chord in enumerate(CHORD_STATES)}
ID_TO_CHORD = {i: chord for chord, i in CHORD_TO_ID.items()}

ROOT_MAP = {'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#'}

RANDOM_SEED = 72

def split_dataset(
    lab_files: List[str], 
    n_train: int = 40, 
    seed: int = RANDOM_SEED
) -> Tuple[List[str], List[str]]:
    
    np.random.seed(seed)
    shuffled_files = list(lab_files)
    np.random.shuffle(shuffled_files)
    
    train_songs = shuffled_files[:n_train]
    test_songs = shuffled_files[n_train:]
    
    print(f"Total songs: {len(shuffled_files)}")
    print(f"Train songs: {len(train_songs)}")
    print(f"Test songs: {len(test_songs)}")
    
    return train_songs, test_songs

def simplify_chord(chord_label):
    if chord_label in ['N', 'X', 'no-chord']:
        return 'N'

    parts = chord_label.split(':')
    root = parts[0]
    root = ROOT_MAP.get(root, root)
    root = root.split('/')[0]

    if root not in CHORD_ROOTS:
        return root

    if len(parts) == 1:
        return root + ':maj'

    t = parts[1]
    t = t.split('/')[0]
    t = t.split('(')[0]
    t = t.replace('aug', '')
    t = t.replace('sus', '')
    t = re.split(r"\d", t)[0]
    t = t.replace('add', '')
    t = t.replace('dim', 'min').replace('hmin', 'min')

    if 'min' in t or t == 'm':
        return root + ':min'
    return root + ':maj'