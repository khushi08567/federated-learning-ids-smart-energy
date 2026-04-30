import sys; sys.stdout.reconfigure(line_buffering=True)
import logging
logging.basicConfig(level=logging.INFO)
def run():
    print("📂 Loading sequence data...")

    # Load both datasets
    X_cic  = np.load("data/processed/X_seq_cic.npy")
    y_cic  = np.load("data/processed/y_seq_cic.npy")
    X_edge = np.load("data/processed/X_seq_edge.npy")
    y_edge = np.load("data/processed/y_seq_edge.npy")

    with open("data/processed/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    num_classes = len(le.classes_)
    print(f"   Classes ({num_classes}): {list(le.classes_)}")

    # ── Combine both datasets ─────────────────────────────────────
    X_all = np.concatenate([X_cic, X_edge], axis=0)
    y_all = np.concatenate([y_cic, y_edge], axis=0)
    print(f"   Combined: {X_all.shape}")

    # ── Remove classes with fewer than 10 samples ─────────────────
    print("\n📊 Class distribution:")
    keep_mask = np.zeros(len(y_all), dtype=bool)
    for cls_idx in range(num_classes):
        count = np.sum(y_all == cls_idx)
        cls_name = le.classes_[cls_idx]
        print(f"   {cls_name:20s}: {count} samples", 
              "⚠️ REMOVED" if count < 10 else "✅")
        if count >= 10:
            keep_mask |= (y_all == cls_idx)

    X_all = X_all[keep_mask]
    y_all = y_all[keep_mask]

    # Re-encode labels to be continuous after removal
    unique_classes = np.unique(y_all)
    remap = {old: new for new, old in enumerate(unique_classes)}
    y_all = np.array([remap[y] for y in y_all])
    num_classes = len(unique_classes)
    kept_class_names = le.classes_[unique_classes]
    print(f"\n   Kept {num_classes} classes: {list(kept_class_names)}")
    print(f"   Final dataset: {X_all.shape}")

    # ── Train/test split (no stratify to avoid issues) ────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.15, random_state=42
    )
    print(f"   Train: {X_train.shape} | Test: {X_test.shape}")

    # Save test set + class info for evaluation
    os.makedirs("saved_models", exist_ok=True)
    np.save("saved_models/X_test.npy", X_test)
    np.save("saved_models/y_test.npy", y_test)
    np.save("saved_models/class_names.npy", kept_class_names)
    print("   ✅ Test set saved")

    # ── Partition into 5 client shards ────────────────────────────
    print(f"\n📊 Partitioning across {NUM_CLIENTS} clients...")
    shards = partition_data(X_train, y_train, NUM_CLIENTS)

    # ── FedAvg strategy ───────────────────────────────────────────
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        on_fit_config_fn=lambda rnd: {
            "local_epochs": 3,
            "batch_size":   64,
            "round":        rnd,
        },
    )

    # ── Start simulation ──────────────────────────────────────────
    print(f"\n🚀 Starting Federated Training ({NUM_ROUNDS} rounds)...")
    history = fl.simulation.start_simulation(
        client_fn=make_client_fn(shards, num_classes),
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 2, "num_gpus": 0.0},
    )

    # ── Save final model ──────────────────────────────────────────
    print("\n💾 Saving final model...")
    n_features  = X_all.shape[2]
    final_model = build_model(WINDOW_SIZE, n_features, num_classes)
    final_model.save("saved_models/fl_model.h5")
    print("✅ Model saved → saved_models/fl_model.h5")

    # ── Print training history ────────────────────────────────────
    print("\n📈 Training History:")
    losses = history.losses_distributed
    for rnd, loss in losses:
        print(f"   Round {rnd:2d} → loss: {loss:.4f}")

    return history