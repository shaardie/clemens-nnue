import json

import torch

NUM_FEATURES = 64 * 64 * 5 * 2 * 2
M = 4
N = 8
K = 1


class NNUE(torch.nn.Module):
    def __init__(self):
        super(NNUE, self).__init__()

        self.l0 = torch.nn.Linear(NUM_FEATURES, M)
        self.l1 = torch.nn.Linear(2 * M, N)
        self.l2 = torch.nn.Linear(N, K)

    # The inputs are a whole batch!
    # `turn` indicates whether white is the side to move. 1 = true, 0 = false.
    def forward(self, white_features, black_features, turn, score, result):
        w = self.l0(white_features)  # white's perspective
        b = self.l0(black_features)  # black's perspective

        # Remember that we order the accumulators for 2 perspectives based on who is to move.
        # So we blend two possible orderings by interpolating between `stm` and `1-stm` tensors.
        accumulator = (turn * torch.cat([w, b], dim=1)) + (
            (1 - turn) * torch.cat([b, w], dim=1)
        )

        # Run the linear layers and use clamp_ as ClippedReLU
        l1_x = torch.clamp(accumulator, 0.0, 1.0)
        l2_x = torch.clamp(self.l1(l1_x), 0.0, 1.0)

        return self.l2(l2_x)

    def save(self, filename):
        d = self.state_dict()
        r = {
            "l0": {"weight": None, "bias": None},
            "l1": {"weight": None, "bias": None},
            "l2": {"weight": None, "bias": None},
        }
        r["l0"]["weight"] = d["l0.weight"].tolist()
        r["l0"]["bias"] = d["l0.bias"].tolist()
        r["l1"]["weight"] = d["l1.weight"].tolist()
        r["l1"]["bias"] = d["l1.bias"].tolist()
        r["l2"]["weight"] = d["l2.weight"].tolist()
        r["l2"]["bias"] = d["l2.bias"].tolist()
        with open(filename, "w") as f:
            json.dump(r, f)
