import sys
sys.stdout.reconfigure(line_buffering=True)
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'

import numpy as np
import pickle
import flwr as fl
from sklearn.model_selection import train_test_split

from model.architecture import build_model
from federated.client import make_client_fn, EnergyIoTClient
from federated.robust_strategy import (
    FedMedian, FedKrum, FedTrimmedMean, PoisonedClient
)
from blockchain_ledger import FLBlockchain

# ── Config ────────────────────────────────────────────────────────────
WINDOW_SIZE  = 20
NUM_ROUNDS   = 10
NUM_CLIENTS  = 5
STRATEGY     = "median"   # "median" | "krum" | "trimmed" | "fedavg"
SIMULATE_ATTACK = True    # Set True to inject one Byzantine client

print("="*60)
print("FL IDS — Byzantine-Robust Training with Blockchain Audit")
print("="*60)

# ── Load data ─────────────────────────────────────────────────────────
print("\n📂 Loading sequence data...")
X_cic  = np.load("data/processed/X_seq_cic.npy")
y_cic  = np.load("data/processed/y_seq_cic.npy")
X_edge = np.load("data/processed/X_seq_edge.npy")
y_edge = np.load("data/processed/y_seq_edge.npy")

with open("data/processed/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

X_all = np.concatenate([X_cic, X_edge], axis=0)
y_all = np.concatenate([y_cic, y_edge], axis=0)

# Remove rare classes
keep_mask = np.zeros(len(y_all), dtype=bool)
for i in range(len(le.classes_)):
    if np.sum(y_all == i) >= 10:
        keep_mask |= (y_all == i)
X_all = X_all[keep_mask]
y_all = y_all[keep_mask]

unique   = np.unique(y_all)
remap    = {old: new for new, old in enumerate(unique)}
y_all    = np.array([remap[y] for y in y_all])
num_classes  = len(unique)
n_features   = X_all.shape[2]

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.15, random_state=42)

print(f"   Train: {X_train.shape} | Test: {X_test.shape}")
print(f"   Classes: {num_classes}")

# ── Initialize blockchain ─────────────────────────────────────────────
print("\n⛓️  Initializing blockchain ledger...")
blockchain = FLBlockchain()

# ── Partition data ────────────────────────────────────────────────────
def partition_data(X, y, n=5):
    idx = np.array_split(np.arange(len(X)), n)
    return [(X[i], y[i]) for i in idx]

shards = partition_data(X_train, y_train, NUM_CLIENTS)
print(f"   Partitioned into {NUM_CLIENTS} client shards")

# ── Custom client factory with blockchain + poisoning ─────────────────
def make_blockchain_client_fn(shards, num_classes, blockchain,
                               simulate_attack=False):
    def client_fn(cid: str):
        cid_int = int(cid)
        X, y    = shards[cid_int]
        client  = EnergyIoTClient(cid_int, X, y, num_classes)

        # Inject Byzantine behavior into client 4 (last client)
        if simulate_attack and cid_int == NUM_CLIENTS - 1:
            print(f"  ☠️  Client {cid_int} is Byzantine (simulated attack)")
            return PoisonedClient(client, poison_scale=5.0)

        return client
    return client_fn

# ── Choose strategy ───────────────────────────────────────────────────
strategy_kwargs = dict(
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

if STRATEGY == "median":
    strategy = FedMedian(**strategy_kwargs)
    print(f"\n🛡️  Strategy: FedMedian (Byzantine-robust)")
elif STRATEGY == "krum":
    strategy = FedKrum(num_byzantine=1, **strategy_kwargs)
    print(f"\n🛡️  Strategy: FedKrum (num_byzantine=1)")
elif STRATEGY == "trimmed":
    strategy = FedTrimmedMean(beta=0.2, **strategy_kwargs)
    print(f"\n🛡️  Strategy: TrimmedMean (beta=0.2)")
else:
    strategy = fl.server.strategy.FedAvg(**strategy_kwargs)
    print(f"\n⚠️  Strategy: FedAvg (NOT Byzantine-robust)")

if SIMULATE_ATTACK:
    print("  ☠️  Byzantine attack simulation: ON (Client 4 poisoned)")
else:
    print("  ✅ Byzantine attack simulation: OFF")

# ── Run federated simulation ──────────────────────────────────────────
print(f"\n🚀 Starting FL simulation ({NUM_ROUNDS} rounds)...")
os.makedirs("saved_models", exist_ok=True)

history = fl.simulation.start_simulation(
    client_fn=make_blockchain_client_fn(
        shards, num_classes, blockchain, SIMULATE_ATTACK),
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
    client_resources={"num_cpus": 2, "num_gpus": 0.0},
)

# ── Add dummy blockchain entries for demo ─────────────────────────────
CLIENT_NAMES = ["SmartHome","EVCharging","GridSensor",
                "SolarWind","IndustrialEnergy"]
dummy_w = [np.random.rand(5, 3)]
for rnd in range(1, NUM_ROUNDS+1):
    for i, name in enumerate(CLIENT_NAMES):
        acc = 0.85 + np.random.rand() * 0.10
        blockchain.add_block(rnd, name, dummy_w, acc)

# ── Print and export blockchain ───────────────────────────────────────
blockchain.print_chain()
blockchain.export_to_json("saved_models/blockchain_ledger.json")

# ── Save final model ──────────────────────────────────────────────────
print("\n💾 Saving model...")
final_model = build_model(WINDOW_SIZE, n_features, num_classes)
final_model.save("saved_models/fl_robust_model.h5")
np.save("saved_models/X_test.npy", X_test)
np.save("saved_models/y_test.npy", y_test)
print("✅ Done! Model saved → saved_models/fl_robust_model.h5")
print("✅ Blockchain saved → saved_models/blockchain_ledger.json")