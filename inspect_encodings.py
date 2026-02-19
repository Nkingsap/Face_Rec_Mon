"""
Utility: Inspect face_encodings.pkl
------------------------------------
Run with:  python inspect_encodings.py
"""

import pickle
import os
import numpy as np

ENCODINGS_FILE = 'encodings/face_encodings.pkl'

if not os.path.exists(ENCODINGS_FILE):
    print(f"[ERROR] File not found: {ENCODINGS_FILE}")
    exit(1)

with open(ENCODINGS_FILE, 'rb') as f:
    data = pickle.load(f)

encodings = data.get('encodings', [])
names     = data.get('names', [])
ids       = data.get('ids', [])

print("=" * 60)
print("  face_encodings.pkl  —  Contents")
print("=" * 60)
print(f"  Total entries : {len(encodings)}")
print(f"  Encoding size : {len(encodings[0]) if encodings else 'N/A'} dimensions")
print("-" * 60)

for i, (name, enc) in enumerate(zip(names, encodings)):
    uid = ids[i] if i < len(ids) else '?'
    norm = float(np.linalg.norm(enc))
    print(f"  [{i+1}]  ID={uid}  Name={name}")
    print(f"       Encoding (first 8 of 128): "
          f"[{', '.join(f'{v:.4f}' for v in enc[:8])} ...]")
    print(f"       Vector norm: {norm:.4f}")
    print()

print("=" * 60)
