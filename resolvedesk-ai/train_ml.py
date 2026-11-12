"""Train the ResolveDesk triage classifier and write results.json."""
import json, os, random, time
from ml import TriageModel, generate_data

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    data = generate_data(n_per_combo=25, seed=42)  # 6 intents x 4 priorities x 25 = 600
    rng = random.Random(7)
    # stratified-ish split: shuffle within each label pair
    train, test = [], []
    buckets = {}
    for d in data:
        buckets.setdefault((d["intent"], d["priority"]), []).append(d)
    for items in buckets.values():
        rng.shuffle(items)
        cut = int(len(items) * 0.8)
        train += items[:cut]; test += items[cut:]
    rng.shuffle(train); rng.shuffle(test)
    t0 = time.time()
    model = TriageModel().train(train)
    secs = time.time() - t0
    metrics = model.evaluate(test)
    model.save(os.path.join(HERE, "triage_model.npz"))
    results = {"dataset_size": len(data), "train_size": len(train),
               "training_seconds": round(secs, 2), **metrics,
               "model_file": "triage_model.npz"}
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
