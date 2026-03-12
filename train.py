#!/usr/bin/env python3
"""
NNUE Training Script

Trains a neural network for chess position evaluation using the NNUE
(Efficiently Updatable Neural Network) architecture.

The network takes a chess position encoded as sparse binary features
(one-hot encoding of piece-type × square for each perspective) and
outputs a win probability that is converted to centipawns.

Architecture:
    Input (768 sparse binary) → Feature Transformer (768 → 256)
    White perspective (256) ─┐
                             ├→ concatenate (512)
    Black perspective (256) ─┘
    → Dense (512 → 32) → Dense (32 → 32) → Output (32 → 1, sigmoid)

All intermediate activations use Clipped ReLU (clamped to [0, 1]).

Usage:
    python train.py
"""

import csv
import os
from typing import List, Tuple
import struct

import chess
import numpy
import math
from lichess_to_csv import OUTPUT as INPUT
from lichess_to_csv import NUMBER_OF_POSITIONS

import torch

# ─── Network Architecture ────────────────────────────────────────
# These constants define the network shape.
# They must match the corresponding Go constants when loading
# the exported weights for inference.

INPUT_SIZE = 768  # 12 piece types × 64 squares
HIDDEN_SIZE = 64  # Feature transformer output width
L1_SIZE = 16  # First hidden layer after concatenation
L2_SIZE = 16  # Second hidden layer
EVAL_SCALE = 400.0  # Sigmoid scaling factor: cp = EVAL_SCALE × raw_output

# ─── Training Hyperparameters ────────────────────────────────────

BATCH_SIZE = 4096
LEARNING_RATE = 0.001
WEIGHT_DECAY = 3e-4  # AdamW weight decay (regularization)
EPOCHS = 100
MAX_PIECES = 32  # Maximum number of pieces on a chess board

# ─── Evaluation Clamping ─────────────────────────────────────────
# Evaluations beyond this range (in centipawns) are clamped.
# This prevents extreme positions from dominating the loss.
CLAMP_SIZE = 2000

# ─── File Paths ──────────────────────────────────────────────────

CACHE_FILE = "dataset_cache.npz"
MODEL_FILE = "nnue.bin"

# ─── Device Selection ────────────────────────────────────────────
# Automatically selects GPU if available, otherwise CPU.
# The entire training pipeline works identically on both devices.

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Piece Type Mapping ──────────────────────────────────────────
# Maps python-chess piece types to contiguous indices 0–5.

PIECE_TYPE_TO_INDEX = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

# Offset to distinguish own pieces (0–5) from opponent pieces (6–11)
PIECE_OFFSET = len(PIECE_TYPE_TO_INDEX)

# Quantization scale for the feature transformer.
# Float values are multiplied by this before storing as int16.
# Must match the Go constant.
QUANT_SCALE = 256

# ─── Data Loading ────────────────────────────────────────────────


def board_to_feature_indices(
    board: chess.Board,
) -> Tuple[List[int], List[int]]:
    """
    Compute active feature indices for both perspectives.

    Each feature encodes (relative_piece_type, square) where:
    - relative_piece_type 0–5 = own pawn..king
    - relative_piece_type 6–11 = opponent pawn..king

    For the white perspective, squares are used as-is (a1=0, h8=63).
    For the black perspective, squares are vertically flipped (XOR 56)
    and own/opponent are swapped.

    Args:
        board: A python-chess Board object.

    Returns:
        (white_indices, black_indices): Lists of active feature indices
        in the range [0, 767].
    """
    white_indices = []
    black_indices = []

    for sq in range(64):
        piece = board.piece_at(sq)
        if piece is None:
            continue

        pt = PIECE_TYPE_TO_INDEX[piece.piece_type]

        # White perspective: own pieces use indices 0–5,
        # opponent pieces use 6–11
        w_pt = pt if piece.color == chess.WHITE else pt + PIECE_OFFSET
        white_indices.append(w_pt * 64 + sq)

        # Black perspective: own/opponent swapped, square flipped
        b_pt = pt if piece.color == chess.BLACK else pt + PIECE_OFFSET
        black_indices.append(b_pt * 64 + (sq ^ 56))

    return white_indices, black_indices


def load_data():
    """
    Load training data from CSV (with caching).

    On first run, reads the CSV file, computes feature indices for
    every position, and saves the results as a compressed .npz file.
    On subsequent runs, loads directly from the cache.

    The data is stored as:
    - stm: (N, MAX_PIECES) int16 array of feature indices for side-to-move
    - opp: (N, MAX_PIECES) int16 array of feature indices for opponent
    - targets: (N,) float32 array of sigmoid-scaled evaluations

    Unused slots in stm/opp are filled with -1.

    Returns:
        (stm, opp, targets) as numpy arrays.
    """
    # Try loading from cache first
    if os.path.exists(CACHE_FILE):
        print(f"Loading cached data from {CACHE_FILE}...")
        data = numpy.load(CACHE_FILE)
        stm = data["stm"]
        opp = data["opp"]
        targets = data["targets"]
        print(f"{len(targets):,} positions loaded from cache.\n")
        return stm, opp, targets

    # Parse CSV and compute features
    stm = numpy.full((NUMBER_OF_POSITIONS, MAX_PIECES), -1, dtype=numpy.int16)
    opp = numpy.full((NUMBER_OF_POSITIONS, MAX_PIECES), -1, dtype=numpy.int16)
    targets = numpy.empty(NUMBER_OF_POSITIONS, dtype=numpy.float32)

    print("Reading positions from CSV...")
    count = 0

    with open(INPUT, "r") as f:
        csv_reader = csv.reader(f)
        for fen, eval_str in csv_reader:
            if count % 10_000 == 0:
                print(f"  {count:,}/{NUMBER_OF_POSITIONS:,}", end="\r")

            board = chess.Board(fen=fen)
            w_idx, b_idx = board_to_feature_indices(board)

            # Skip positions with more than 32 pieces (irregular)
            if len(w_idx) > MAX_PIECES:
                continue

            # Clamp evaluation to prevent extreme outliers
            eval_cp = max(min(int(eval_str), CLAMP_SIZE), -CLAMP_SIZE)

            # Normalize to side-to-move perspective:
            # The evaluation in the CSV is from white's perspective.
            # If black is to move, we negate it so that positive values
            # always mean "good for the side to move".
            if board.turn == chess.BLACK:
                eval_cp *= -1
                stm_idx, opp_idx = b_idx, w_idx
            else:
                stm_idx, opp_idx = w_idx, b_idx

            stm[count, : len(stm_idx)] = stm_idx
            opp[count, : len(opp_idx)] = opp_idx

            # Convert centipawns to win probability using sigmoid:
            # target = sigmoid(cp / EVAL_SCALE)
            # This compresses the evaluation range so that extreme
            # values (+5000 cp) don't dominate the loss function.
            targets[count] = 1.0 / (1.0 + math.exp(-eval_cp / EVAL_SCALE))
            count += 1

    # Trim arrays to actual number of positions
    stm = stm[:count]
    opp = opp[:count]
    targets = targets[:count]
    print(f"{count:,} positions loaded.")

    # Save cache for next time
    print(f"Saving cache to {CACHE_FILE}...")
    numpy.savez_compressed(CACHE_FILE, stm=stm, opp=opp, targets=targets)
    print("Cache saved.\n")

    return stm, opp, targets


# ─── Model Definition ────────────────────────────────────────────


class NNUEModel(torch.nn.Module):
    """
    NNUE-style network for chess position evaluation.

    Architecture:
        Feature Transformer (shared weights, applied to both perspectives):
            Linear(768 → HIDDEN_SIZE) + Clipped ReLU

        The outputs for both perspectives are concatenated (STM first,
        then opponent), giving a vector of size HIDDEN_SIZE × 2.

        This is followed by small dense layers:
            Linear(HIDDEN_SIZE×2 → L1_SIZE) + Clipped ReLU
            Linear(L1_SIZE → L2_SIZE) + Clipped ReLU
            Linear(L2_SIZE → 1) + Sigmoid

    The output is a win probability in [0, 1].
    To convert back to centipawns: cp ≈ EVAL_SCALE × raw_output
    (where raw_output is the value before the final sigmoid).
    """

    def __init__(self):
        super().__init__()
        self.ft = torch.nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.l1 = torch.nn.Linear(HIDDEN_SIZE * 2, L1_SIZE)
        self.l2 = torch.nn.Linear(L1_SIZE, L2_SIZE)
        self.out = torch.nn.Linear(L2_SIZE, 1)

    def forward(self, stm_features, opp_features):
        # Feature transformer: same weights for both perspectives
        stm_h = torch.clamp(self.ft(stm_features), 0.0, 1.0)
        opp_h = torch.clamp(self.ft(opp_features), 0.0, 1.0)

        # Concatenate: side-to-move first, then opponent
        x = torch.cat([stm_h, opp_h], dim=1)

        # Dense layers with clipped ReLU
        x = torch.clamp(self.l1(x), 0.0, 1.0)
        x = torch.clamp(self.l2(x), 0.0, 1.0)

        # Output: sigmoid squashes to [0, 1] (win probability)
        x = torch.sigmoid(self.out(x))
        return x


# ─── Sparse to Dense Conversion ──────────────────────────────────


def indices_to_dense(indices, batch_size):
    """
    Convert sparse feature indices to a dense binary tensor.

    Takes a (batch_size × MAX_PIECES) tensor of int16 feature indices
    (with -1 as padding for empty slots) and returns a
    (batch_size × INPUT_SIZE) float32 tensor with 1.0 at each
    active feature index and 0.0 elsewhere.

    This runs entirely on whatever device the input tensor is on
    (GPU or CPU), avoiding any data transfer.

    Args:
        indices: (batch_size, MAX_PIECES) int16 tensor on DEVICE.
                 Values in [0, 767] are active features, -1 is padding.
        batch_size: Number of samples in this batch.

    Returns:
        (batch_size, INPUT_SIZE) float32 tensor on DEVICE.
    """
    feat = torch.zeros(batch_size, INPUT_SIZE, device=indices.device)

    # Build row indices matching the shape of `indices`:
    # [[0, 0, ..., 0],
    #  [1, 1, ..., 1],
    #  ...]
    row_idx = (
        torch.arange(batch_size, device=indices.device).unsqueeze(1).expand_as(indices)
    )

    # Only set entries where the index is valid (not -1)
    mask = indices >= 0
    feat[row_idx[mask], indices[mask].long()] = 1.0

    return feat


# ─── Training Loop ────────────────────────────────────────────────


def train():
    """
    Main training function.

    Loads all data onto the selected device (GPU or CPU), splits into
    train/validation sets, and trains the NNUE model with manual
    batching. No DataLoader is used — this avoids CPU↔GPU data
    transfer overhead and shared memory issues.

    The best model (by validation loss) is saved to MODEL_FILE.
    """
    stm_np, opp_np, targets_np = load_data()

    # Move everything to the target device (GPU VRAM or CPU RAM).
    # For 30M positions this uses ~5 GB, well within 12 GB GPU memory.
    print(f"Moving data to {DEVICE}...")
    stm_dev = torch.tensor(stm_np, dtype=torch.int16, device=DEVICE)
    opp_dev = torch.tensor(opp_np, dtype=torch.int16, device=DEVICE)
    targets_dev = torch.tensor(targets_np, dtype=torch.float32, device=DEVICE)

    # Free CPU memory
    del stm_np, opp_np, targets_np

    # Train/validation split (90/10)
    n = len(targets_dev)
    n_train = int(0.9 * n)
    n_val = n - n_train

    perm = torch.randperm(n, device=DEVICE)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    # Model, optimizer, scheduler
    model = NNUEModel().to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )
    loss_fn = torch.nn.MSELoss()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU memory used: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    print(f"Parameters: {param_count:,}")
    print(f"Training positions: {n_train:,}")
    print(f"Validation positions: {n_val:,}\n")

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        # ── Training ─────────────────────────────────────────────
        model.train()
        t_loss, t_batches = 0.0, 0

        # Shuffle training indices each epoch
        shuffled = train_idx[torch.randperm(n_train, device=DEVICE)]

        for start in range(0, n_train, BATCH_SIZE):
            idx = shuffled[start : start + BATCH_SIZE]
            bs = len(idx)

            stm = indices_to_dense(stm_dev[idx], bs)
            opp = indices_to_dense(opp_dev[idx], bs)
            target = targets_dev[idx].unsqueeze(1)

            pred = model(stm, opp)
            loss = loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t_loss += loss.item()
            t_batches += 1

        # ── Validation ───────────────────────────────────────────
        model.eval()
        v_loss, v_batches = 0.0, 0

        with torch.no_grad():
            for start in range(0, n_val, BATCH_SIZE):
                idx = val_idx[start : start + BATCH_SIZE]
                bs = len(idx)

                stm = indices_to_dense(stm_dev[idx], bs)
                opp = indices_to_dense(opp_dev[idx], bs)
                target = targets_dev[idx].unsqueeze(1)

                pred = model(stm, opp)
                v_loss += loss_fn(pred, target).item()
                v_batches += 1

        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        avg_val_loss = v_loss / v_batches

        # Save model if validation loss improved
        marker = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_weights(model, MODEL_FILE)
            marker = " ← saved"

        print(
            f"Epoch {epoch + 1:2d}/{EPOCHS} │ "
            f"train loss: {t_loss / t_batches:.6f} │ "
            f"val loss:   {avg_val_loss:.6f} │ "
            f"LR: {lr:.6f}{marker}"
        )

    print(f"\nBest model (val loss {best_val_loss:.6f}) saved to {MODEL_FILE}")


# ─── Weight Export ────────────────────────────────────────────────


def save_weights(model: NNUEModel, path: str):
    """
    Export model weights as flat little-endian float32 values.

    This binary format is designed for easy loading in Go without
    any ML framework dependencies.

    File layout:
        1. Header: 4 × uint32 (INPUT_SIZE, HIDDEN_SIZE, L1_SIZE, L2_SIZE)
        2. FT weights  (INPUT_SIZE × HIDDEN_SIZE float32, TRANSPOSED)
           Transposed so that each feature's weight vector is contiguous
           in memory — this enables efficient incremental updates.
        3. FT bias     (HIDDEN_SIZE float32)
        4. L1 weights  (L1_SIZE × HIDDEN_SIZE*2 float32, row-major)
        5. L1 bias     (L1_SIZE float32)
        6. L2 weights  (L2_SIZE × L1_SIZE float32, row-major)
        7. L2 bias     (L2_SIZE float32)
        8. Out weights (L2_SIZE float32)
        9. Out bias    (1 float32)

    Args:
        model: Trained NNUEModel instance.
        path: Output file path.
    """
    with open(path, "wb") as f:
        # Header with architecture dimensions for validation on load
        for v in (INPUT_SIZE, HIDDEN_SIZE, L1_SIZE, L2_SIZE):
            f.write(struct.pack("<I", v))

        def write_tensor(t):
            arr = t.detach().cpu().numpy().astype(numpy.float32)
            f.write(arr.tobytes())

        # Feature transformer weights: transposed from (HIDDEN, INPUT)
        # to (INPUT, HIDDEN) so that feature[i] is a contiguous block.
        write_tensor(model.ft.weight.t())
        write_tensor(model.ft.bias)

        # Remaining layers: stored in PyTorch's default (out × in) layout
        write_tensor(model.l1.weight)
        write_tensor(model.l1.bias)
        write_tensor(model.l2.weight)
        write_tensor(model.l2.bias)
        write_tensor(model.out.weight)
        write_tensor(model.out.bias)


if __name__ == "__main__":
    train()
