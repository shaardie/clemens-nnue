#!/usr/bin/env python3

import csv
from typing import List, Tuple
import struct
import os.path

import chess
import numpy
import math
from lichess_to_csv import OUTPUT as INPUT
from lichess_to_csv import NUMBER_OF_POSITIONS

from torch.utils.data import Dataset, DataLoader
import torch

OUTPUT = "features.csv"
CLAMP_SIZE = 2000

INPUT_SIZE = 768  # 12 Figurentypen × 64 Felder
HIDDEN_SIZE = 256  # Feature-Transformer-Breite
L1_SIZE = 32  # Erste versteckte Schicht
L2_SIZE = 32  # Zweite versteckte Schicht
EVAL_SCALE = 400.0  # Sigmoid-Skalierung

BATCH_SIZE = 4096
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
EPOCHS = 50
MAX_PIECES = 32  # Maximal 32 Figuren auf dem Brett

CACHE_FILE = "dataset_cache.npz"

PIECE_TYPE_TO_NUMBER = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

PIECE_OFFSET = len(PIECE_TYPE_TO_NUMBER)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ChessDataset(Dataset):
    def __init__(self) -> None:
        if os.path.exists(CACHE_FILE):
            print("load data from cache")
            data = numpy.load(CACHE_FILE)
            self.stm = data["stm"]
            self.opp = data["opp"]
            self.targets = data["targets"]
            print(f"{len(self.targets)} positions read from cache")
            return

        with open(INPUT, "r") as input:
            csv_reader = csv.reader(input)

            self.stm = numpy.full(
                (NUMBER_OF_POSITIONS, MAX_PIECES), -1, dtype=numpy.int16
            )
            self.opp = numpy.full(
                (NUMBER_OF_POSITIONS, MAX_PIECES), -1, dtype=numpy.int16
            )
            self.targets = numpy.empty(NUMBER_OF_POSITIONS, dtype=numpy.float32)

            print("Read positions")
            number_of_positions = 0
            max_number_of_positions = NUMBER_OF_POSITIONS
            for fen, eval in csv_reader:
                if number_of_positions % 100_00 == 0:
                    print(
                        f"{number_of_positions}/{max_number_of_positions} completed",
                        end="\r",
                    )
                board = chess.Board(fen=fen)
                white_indices, black_indices = self._board_to_feature_indices(board)
                # strange positions with more pieces than usual
                # we do not train on them
                if len(white_indices) > 32:
                    max_number_of_positions -= 1
                    continue
                eval = max(min(int(eval), CLAMP_SIZE), -CLAMP_SIZE)
                if board.turn == chess.BLACK:
                    eval *= -1
                    stm_indices = black_indices
                    opp_indices = white_indices
                else:
                    stm_indices = white_indices
                    opp_indices = black_indices

                self.stm[number_of_positions, : len(stm_indices)] = stm_indices
                self.opp[number_of_positions, : len(opp_indices)] = opp_indices
                self.targets[number_of_positions] = 1.0 / (
                    1.0 + math.exp(-eval / EVAL_SCALE)
                )
                number_of_positions += 1

            self.stm = self.stm[:number_of_positions]
            self.opp = self.opp[:number_of_positions]
            self.targets = self.targets[:number_of_positions]

            print(f"\n{number_of_positions} positions read")

        print("save positions to cache")
        numpy.savez_compressed(
            CACHE_FILE, stm=self.stm, opp=self.opp, targets=self.targets
        )
        print("cache saved")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # Sparse → Dense
        stm_feat = torch.zeros(INPUT_SIZE)
        active = self.stm[idx]
        active = active[active >= 0]
        stm_feat[active.astype(numpy.int64)] = 1.0

        opp_feat = torch.zeros(INPUT_SIZE)
        active = self.opp[idx]
        active = active[active >= 0]
        opp_feat[active.astype(numpy.int64)] = 1.0

        return stm_feat, opp_feat, self.targets[idx]

    def _board_to_feature_indices(
        self,
        board: chess.Board,
    ) -> Tuple[List[int], List[int]]:
        """
        Calculates the active features idx for white and black
        """
        white_indices = []
        black_indices = []

        for sq in range(64):
            piece = board.piece_at(sq)
            if piece is None:
                continue

            idx = PIECE_TYPE_TO_NUMBER[piece.piece_type]

            # feature from whites perspective
            white_idx = idx if piece.color == chess.WHITE else idx + PIECE_OFFSET
            white_idx = white_idx * 64 + sq
            white_indices.append(white_idx)

            # feature from blacks perspective
            black_idx = idx if piece.color == chess.BLACK else idx + PIECE_OFFSET
            black_idx = black_idx * 64 + (sq ^ 56)  # flip square
            black_indices.append(black_idx)

        return white_indices, black_indices


class NNUEModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ft = torch.nn.Linear(INPUT_SIZE, HIDDEN_SIZE)  # 768 → 256
        self.l1 = torch.nn.Linear(HIDDEN_SIZE * 2, L1_SIZE)  # 512 → 32
        self.l2 = torch.nn.Linear(L1_SIZE, L2_SIZE)  # 32  → 32
        self.out = torch.nn.Linear(L2_SIZE, 1)  # 32  → 1

    def forward(self, stm_features, opp_features):
        # Feature-Transformer (gleiche Gewichte für beide Perspektiven)
        stm_h = torch.clamp(self.ft(stm_features), 0.0, 1.0)
        opp_h = torch.clamp(self.ft(opp_features), 0.0, 1.0)

        # Zusammenführen: Seite am Zug zuerst
        x = torch.cat([stm_h, opp_h], dim=1)

        x = torch.clamp(self.l1(x), 0.0, 1.0)
        x = torch.clamp(self.l2(x), 0.0, 1.0)
        x = torch.sigmoid(self.out(x))
        return x


def train():
    dataset = ChessDataset()

    # 90 / 10 Split
    n_train = int(0.9 * len(dataset))
    n_val = len(dataset) - n_train
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=(DEVICE == "cuda"),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=(DEVICE == "cuda"),
    )

    model = NNUEModel().to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    loss_fn = torch.nn.MSELoss()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Device: {DEVICE}")
    print(f"Parameter: {param_count:,}")
    print(f"tranings positions: {n_train:,}")
    print(f"validation positions: {n_val:,}\n")

    for epoch in range(EPOCHS):
        # --- training ---
        model.train()
        t_loss, t_batches = 0.0, 0

        for stm, opp, target in train_loader:
            stm = stm.to(DEVICE)
            opp = opp.to(DEVICE)
            target = target.to(DEVICE).unsqueeze(1)

            pred = model(stm, opp)
            loss = loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t_loss += loss.item()
            t_batches += 1

        # --- validation ---
        model.eval()
        v_loss, v_batches = 0.0, 0

        with torch.no_grad():
            for stm, opp, target in val_loader:
                stm = stm.to(DEVICE)
                opp = opp.to(DEVICE)
                target = target.to(DEVICE).unsqueeze(1)

                pred = model(stm, opp)
                v_loss += loss_fn(pred, target).item()
                v_batches += 1

        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch + 1:2d}/{EPOCHS} │ "
            f"train loss: {t_loss / t_batches:.6f} │ "
            f"validation loss:   {v_loss / v_batches:.6f} │ "
            f"LR: {lr:.6f}"
        )

    # save wheights
    save_weights(model, "nnue.bin")
    print(f"\nmodell saved")


# ─── Export wheights ────────────────────────────────────────


def save_weights(model: NNUEModel, path: str):
    """
    Speichert alle Gewichte als flache Float32-Werte (Little-Endian).

    Reihenfolge:
        1. Header: 4 × uint32 (input_size, hidden_size, l1_size, l2_size)
        2. FT weights  (768×256 float32, TRANSPONIERT)
        3. FT bias     (256 float32)
        4. L1 weights  (32×512 float32, out×in)
        5. L1 bias     (32 float32)
        6. L2 weights  (32×32 float32)
        7. L2 bias     (32 float32)
        8. Out weights (1×32 float32)
        9. Out bias    (1 float32)
    """
    with open(path, "wb") as f:
        # Header
        for v in (INPUT_SIZE, HIDDEN_SIZE, L1_SIZE, L2_SIZE):
            f.write(struct.pack("<I", v))

        def write_tensor(t):
            arr = t.detach().cpu().numpy().astype(numpy.float32)
            f.write(arr.tobytes())

        # FT: transponiert speichern → (768, 256) statt (256, 768)
        # So liegt jedes Feature als zusammenhängender 256er-Block im Speicher
        write_tensor(model.ft.weight.t())
        write_tensor(model.ft.bias)

        # Rest: PyTorch-Standard (out × in)
        write_tensor(model.l1.weight)
        write_tensor(model.l1.bias)
        write_tensor(model.l2.weight)
        write_tensor(model.l2.bias)
        write_tensor(model.out.weight)
        write_tensor(model.out.bias)


if __name__ == "__main__":
    train()
