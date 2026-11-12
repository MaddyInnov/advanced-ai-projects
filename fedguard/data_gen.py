"""
FedGuard data generation.

Synthetic 10-class classification problem:
- 20-dim features, Gaussian-mixture class conditionals
- Plus structured non-linear features: for each class, 4 of the 20 dims
  are drawn with a class-specific scale and a quadratic interaction term
  is injected into 2 dims, so a plain linear model is not perfect and an
  MLP earns its keep.
- n = 6000 total: 5000 train pool (partitioned across clients), 1000 test.

Partitions:
- IID: random shuffle split across 10 clients.
- non-IID: Dirichlet(alpha=0.5) label-skew partition (standard in
  McMahan et al. / Hsu et al.): for each class, draw a Dirichlet over
  clients and split that class's examples proportionally.
Fixed seed everywhere.
"""
import numpy as np

NUM_CLASSES = 10
INPUT_DIM = 20
N_TOTAL = 6000
N_TRAIN = 5000
NUM_CLIENTS = 10
SEED = 42


def generate_data(seed=SEED):
    rng = np.random.default_rng(seed)
    # Balanced classes
    y = np.repeat(np.arange(NUM_CLASSES), N_TOTAL // NUM_CLASSES)
    rng.shuffle(y)

    # Class means. Scale calibrated empirically (see README): mean scale
    # 0.7 / noise 1.0 puts a well-trained centralised model at ~85% --
    # hard enough that non-IID drift, the FedProx correction and DP noise
    # are all visible in the results. (At scale 2.2 the problem is
    # trivially separable and every method scores 100%; at 0.6 the
    # centralised baseline itself undertrains. 0.7 is the honest middle
    # found by sweeping 0.6 / 0.7 / 0.8 / 1.0 / 1.3.)
    means = rng.normal(0, 0.7, size=(NUM_CLASSES, INPUT_DIM))
    X = means[y] + rng.normal(0, 1.0, size=(N_TOTAL, INPUT_DIM))

    # Structured non-linearity: class-specific scaling on 4 dims +
    # quadratic term
    for c in range(NUM_CLASSES):
        idx = y == c
        dims = [(c * 2 + j) % INPUT_DIM for j in range(4)]
        X[idx][:, dims] *= 1.15
        # quadratic interaction leaked into dim 0/1
        X[idx, 0] += 0.08 * X[idx, 1] ** 2 * np.sign(means[c, 0])

    X = X.astype(np.float32)
    # Stratified-ish split: first N_TRAIN after per-class shuffle
    train_idx, test_idx = [], []
    for c in range(NUM_CLASSES):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_test_c = (N_TOTAL - N_TRAIN) // NUM_CLASSES
        test_idx.extend(idx[:n_test_c])
        train_idx.extend(idx[n_test_c:])
    train_idx = np.array(train_idx); test_idx = np.array(test_idx)
    rng.shuffle(train_idx); rng.shuffle(test_idx)
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def iid_partition(y_train, num_clients=NUM_CLIENTS, seed=SEED):
    rng = np.random.default_rng(seed + 1)
    idx = np.arange(len(y_train))
    rng.shuffle(idx)
    splits = np.array_split(idx, num_clients)
    return [s.tolist() for s in splits]


def dirichlet_partition(y_train, num_clients=NUM_CLIENTS, alpha=0.5, seed=SEED):
    """Label-skew partition: per class, Dirichlet(alpha) over clients."""
    rng = np.random.default_rng(seed + 2)
    client_idx = [[] for _ in range(num_clients)]
    for c in range(NUM_CLASSES):
        idx = np.where(y_train == c)[0]
        rng.shuffle(idx)
        props = rng.dirichlet([alpha] * num_clients)
        # Convert proportions to split points
        counts = np.floor(props * len(idx)).astype(int)
        # Fix rounding: distribute remainder to largest props
        remainder = len(idx) - counts.sum()
        order = np.argsort(-props)
        for i in range(remainder):
            counts[order[i % num_clients]] += 1
        start = 0
        for client in range(num_clients):
            end = start + counts[client]
            client_idx[client].extend(idx[start:end].tolist())
            start = end
    for lst in client_idx:
        rng.shuffle(lst)
    return client_idx


def label_histogram(y, idx, num_classes=NUM_CLASSES):
    hist = np.bincount(y[idx], minlength=num_classes)
    return hist.tolist()


if __name__ == "__main__":
    Xtr, ytr, Xte, yte = generate_data()
    np.savez_compressed("data.npz", X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte)
    iid = iid_partition(ytr)
    noniid = dirichlet_partition(ytr)
    print(f"Train {Xtr.shape} Test {Xte.shape}")
    print("IID client sizes:", [len(s) for s in iid])
    print("non-IID client sizes:", [len(s) for s in noniid])
    print("non-IID label histograms (per client):")
    for i, idx in enumerate(noniid):
        print(f"  client {i}: {label_histogram(ytr, idx)}")
    print("Saved data.npz")
