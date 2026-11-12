import torch
import torch.nn as nn

INPUT_DIM = 20
NUM_CLASSES = 10
DROPOUT = 0.3


class MLP(nn.Module):
    """20 -> 64 -> 64 -> 10 MLP with dropout (p=0.3).

    Dropout is part of the shared model for every method in this project
    (centralised, FedAvg, FedProx, DP-SGD): during design-time calibration
    the plain MLP overfit the pooled data (train 96.8% / test 88.0%),
    while dropout 0.3 shrank the gap (train ~93% / test ~89%) and made
    the centralised-vs-federated comparison fair. Note: the DP-SGD path
    computes per-sample gradients in eval mode, so dropout is inactive
    there (documented in README).
    """

    def __init__(self, input_dim=INPUT_DIM, hidden=64, num_classes=NUM_CLASSES,
                 dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def get_state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def set_state(model, state):
    model.load_state_dict({k: v.clone() for k, v in state.items()})


if __name__ == "__main__":
    m = MLP()
    print(f"MLP params: {count_params(m):,}")
    x = torch.randn(4, INPUT_DIM)
    print("Output shape:", m(x).shape)
