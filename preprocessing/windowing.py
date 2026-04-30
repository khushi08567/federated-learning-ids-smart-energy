import numpy as np
import os

SAVE_DIR = "data/processed/"
WINDOW   = 20
STRIDE   = 5

def create_sequences(X, y, window=20, stride=5):
    Xs, ys = [], []
    total = (len(X) - window) // stride
    print(f"   Total sequences to create: ~{total}")
    
    for count, i in enumerate(range(0, len(X) - window, stride)):
        Xs.append(X[i : i + window])
        labels_in_window = y[i : i + window]
        ys.append(np.bincount(labels_in_window).argmax())
        
        # Print progress every 10000 sequences
        if (count + 1) % 10000 == 0:
            pct = (count + 1) / total * 100
            print(f"   Progress: {count+1}/{total} ({pct:.1f}%)")

    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int32)

def process_dataset(name):
    print(f"\n📂 Processing {name}...")
    X = np.load(f"{SAVE_DIR}X_{name}.npy")
    y = np.load(f"{SAVE_DIR}y_{name}.npy")
    print(f"   Loaded X:{X.shape}  y:{y.shape}")

    # Sample to avoid memory issues
    MAX_ROWS = 200_000
    if len(X) > MAX_ROWS:
        idx = np.random.choice(len(X), MAX_ROWS, replace=False)
        idx = np.sort(idx)
        X, y = X[idx], y[idx]
        print(f"   Sampled down to {MAX_ROWS} rows")

    Xs, ys = create_sequences(X, y, window=WINDOW, stride=STRIDE)
    print(f"   Sequences → X:{Xs.shape}  y:{ys.shape}")

    np.save(f"{SAVE_DIR}X_seq_{name}.npy", Xs)
    np.save(f"{SAVE_DIR}y_seq_{name}.npy", ys)
    print(f"   ✅ Saved X_seq_{name}.npy and y_seq_{name}.npy")
    return Xs, ys

if __name__ == "__main__":
    print("🔄 Creating sliding window sequences...")
    print(f"   Window size : {WINDOW}")
    print(f"   Stride      : {STRIDE}")

    Xs_cic,  ys_cic  = process_dataset("cic")
    Xs_edge, ys_edge = process_dataset("edge")

    print(f"\n✅ ALL DONE!")
    print(f"   CICIoT   sequences → {Xs_cic.shape}")
    print(f"   EdgeIIoT sequences → {Xs_edge.shape}")