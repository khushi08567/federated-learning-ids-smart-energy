import hashlib
import json
import time
import numpy as np

class Block:
    def __init__(self, index, round_num, client_id,
                 weight_hash, accuracy, previous_hash):
        self.index         = index
        self.timestamp     = time.strftime("%Y-%m-%d %H:%M:%S")
        self.round_num     = round_num
        self.client_id     = client_id
        self.weight_hash   = weight_hash
        self.accuracy      = accuracy
        self.previous_hash = previous_hash
        self.hash          = self.compute_hash()

    def compute_hash(self):
        block_string = json.dumps({
            "index":         self.index,
            "timestamp":     self.timestamp,
            "round_num":     self.round_num,
            "client_id":     self.client_id,
            "weight_hash":   self.weight_hash,
            "accuracy":      self.accuracy,
            "previous_hash": self.previous_hash,
        }, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()


class FLBlockchain:
    """
    Lightweight blockchain to audit federated learning
    weight updates. Each client weight submission per
    round is recorded as a tamper-proof block.
    """
    def __init__(self):
        self.chain = []
        self._create_genesis_block()

    def _create_genesis_block(self):
        genesis = Block(
            index=0, round_num=0,
            client_id="GENESIS",
            weight_hash="0" * 64,
            accuracy=0.0,
            previous_hash="0" * 64,
        )
        self.chain.append(genesis)
        print("⛓️  Blockchain initialized with genesis block")

    @property
    def last_block(self):
        return self.chain[-1]

    def hash_weights(self, weights: list) -> str:
        """Create SHA-256 hash of model weight arrays."""
        weight_bytes = b""
        for w in weights:
            weight_bytes += np.array(w).tobytes()
        return hashlib.sha256(weight_bytes).hexdigest()

    def add_block(self, round_num: int, client_id: str,
                  weights: list, accuracy: float) -> Block:
        """
        Record a client's weight submission for a given round.
        Returns the new block.
        """
        weight_hash = self.hash_weights(weights)
        block = Block(
            index         = len(self.chain),
            round_num     = round_num,
            client_id     = client_id,
            weight_hash   = weight_hash,
            accuracy      = round(accuracy, 4),
            previous_hash = self.last_block.hash,
        )
        self.chain.append(block)
        return block

    def is_chain_valid(self) -> bool:
        """Verify the entire chain has not been tampered with."""
        for i in range(1, len(self.chain)):
            current  = self.chain[i]
            previous = self.chain[i - 1]
            if current.hash != current.compute_hash():
                print(f"❌ Block {i} hash is invalid!")
                return False
            if current.previous_hash != previous.hash:
                print(f"❌ Block {i} previous_hash mismatch!")
                return False
        return True

    def verify_weight_integrity(self, round_num: int,
                                client_id: str,
                                weights: list) -> bool:
        """
        Check if submitted weights match what was recorded
        on the blockchain for a specific round + client.
        """
        submitted_hash = self.hash_weights(weights)
        for block in self.chain:
            if (block.round_num == round_num and
                    block.client_id == client_id):
                if block.weight_hash == submitted_hash:
                    return True
                else:
                    print(f"⚠️  Weight tampering detected! "
                          f"Client {client_id} Round {round_num}")
                    return False
        return False

    def get_round_summary(self, round_num: int) -> dict:
        """Get all blocks for a specific FL round."""
        blocks = [b for b in self.chain if b.round_num == round_num]
        return {
            "round":    round_num,
            "clients":  len(blocks),
            "blocks":   [
                {
                    "client_id":   b.client_id,
                    "accuracy":    b.accuracy,
                    "weight_hash": b.weight_hash[:16] + "...",
                    "timestamp":   b.timestamp,
                    "block_hash":  b.hash[:16] + "...",
                }
                for b in blocks
            ],
        }

    def print_chain(self):
        """Print the full blockchain in readable format."""
        print("\n" + "="*60)
        print("⛓️  FEDERATED LEARNING BLOCKCHAIN LEDGER")
        print("="*60)
        for block in self.chain:
            print(f"\nBlock #{block.index}")
            print(f"  Round      : {block.round_num}")
            print(f"  Client     : {block.client_id}")
            print(f"  Accuracy   : {block.accuracy:.4f}")
            print(f"  Time       : {block.timestamp}")
            print(f"  WeightHash : {block.weight_hash[:24]}...")
            print(f"  BlockHash  : {block.hash[:24]}...")
            print(f"  PrevHash   : {block.previous_hash[:24]}...")
        print("\n" + "="*60)
        valid = self.is_chain_valid()
        print(f"Chain integrity: {'✅ VALID' if valid else '❌ COMPROMISED'}")
        print("="*60 + "\n")

    def export_to_json(self, path="blockchain_ledger.json"):
        """Save blockchain to JSON file."""
        data = [
            {
                "index":         b.index,
                "timestamp":     b.timestamp,
                "round_num":     b.round_num,
                "client_id":     b.client_id,
                "weight_hash":   b.weight_hash,
                "accuracy":      b.accuracy,
                "previous_hash": b.previous_hash,
                "hash":          b.hash,
            }
            for b in self.chain
        ]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"✅ Blockchain exported → {path}")


if __name__ == "__main__":
    # Demo test
    bc = FLBlockchain()
    dummy_weights = [np.random.rand(10, 5), np.random.rand(5)]
    bc.add_block(1, "SmartHome",          dummy_weights, 0.91)
    bc.add_block(1, "EVCharging",         dummy_weights, 0.89)
    bc.add_block(1, "GridSensor",         dummy_weights, 0.92)
    bc.add_block(2, "SmartHome",          dummy_weights, 0.93)
    bc.add_block(2, "SolarWind",          dummy_weights, 0.94)
    bc.print_chain()
    bc.export_to_json()