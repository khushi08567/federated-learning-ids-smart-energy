import sys
sys.stdout.reconfigure(line_buffering=True)

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'

import numpy as np
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from model.architecture import build_model

print("=" * 50)
print("STEP 1: Loading data...")

X_cic  = np.load("data/processed/X_seq_cic.npy")
y_cic  = np.load("data/processed/y_seq_cic.npy")
X_edge = np.load("data/processed/X_seq_edge.npy")
y_edge = np.load("data/processed/y_seq_edge.npy")

print(f"CICIoT  : {X_cic.shape}")
print(f"EdgeIIoT: {X_edge.shape}")

with open("data/processed/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

print("=" * 50)
print("STEP 2: Combining datasets...")

X_all = np.concatenate([X_cic, X_edge], axis=0)
y_all = np.concatenate([y_cic, y_edge], axis=0)
print(f"Combined: {X_all.shape}")

# Remove rare classes
print("\nClass distribution:")
keep_mask = np.zeros(len(y_all), dtype=bool)
for i in range(len(le.classes_)):
    count = np.sum(y_all == i)
    status = "✅ KEEP" if count >= 10 else "❌ REMOVE"
    print(f"  {le.classes_[i]:20s}: {count:6d}  {status}")
    if count >= 10:
        keep_mask |= (y_all == i)

X_all = X_all[keep_mask]
y_all = y_all[keep_mask]

# Remap labels
unique = np.unique(y_all)
remap  = {old: new for new, old in enumerate(unique)}
y_all  = np.array([remap[y] for y in y_all])
num_classes = len(unique)
class_names = le.classes_[unique]
print(f"\nKept {num_classes} classes: {list(class_names)}")

print("=" * 50)
print("STEP 3: Train/test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42
)
print(f"Train: {X_train.shape}")
print(f"Test : {X_test.shape}")

print("=" * 50)
print("STEP 4: Building model...")

WINDOW    = X_all.shape[1]
FEATURES  = X_all.shape[2]
model = build_model(WINDOW, FEATURES, num_classes)
print(f"Model built: input({WINDOW},{FEATURES}) → output({num_classes})")

print("=" * 50)
print("STEP 5: Training...")

os.makedirs("saved_models", exist_ok=True)

y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
y_test_cat  = tf.keras.utils.to_categorical(y_test,  num_classes)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        patience=3, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.5, patience=2, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(
        "saved_models/best_model.h5",
        save_best_only=True, verbose=1),
]

# Verify model is not None
print(f"Model type: {type(model)}")
print(f"Num classes: {num_classes}")
print(f"y_train_cat shape: {y_train_cat.shape}")

history = model.fit(
    X_train, y_train_cat,
    epochs=20,
    batch_size=64,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1,
)

print("=" * 50)
print("STEP 6: Evaluating...")
loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"Test Accuracy : {acc:.4f}")
print(f"Test Loss     : {loss:.4f}")

print("=" * 50)
print("STEP 7: Saving...")
model.save("saved_models/fl_model.h5")
np.save("saved_models/X_test.npy", X_test)
np.save("saved_models/y_test.npy", y_test)
np.save("saved_models/class_names.npy", class_names)

print("✅ Model saved → saved_models/fl_model.h5")
print("✅ Test data   → saved_models/X_test.npy")
print("✅ Classes     → saved_models/class_names.npy")
print("\n🎉 TRAINING COMPLETE!")