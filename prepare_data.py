"""
Lichess Eval DB -> HalfKP Binary Training Data
===============================================
Reads lichess_db_eval.jsonl.zst and writes a compact binary training format.

Usage:
    pip install python-chess zstandard numpy
    python prepare_data.py --input lichess_db_eval.jsonl.zst --output train.bin --limit 5000000

Output format (per sample, little-endian):
    - int16:   cp score (clipped to +-2000, from the perspective of the side to move)
    - uint8:   number of active features side A (white king POV)
    - uint8:   number of active features side B (black king POV)
    - uint16 x n_a: feature indices side A
    - uint16 x n_b: feature indices side B

The format is variable-length but very compact (~300 bytes/sample typical).
An index file is also written for fast random access during training.
"""

import argparse
import json
import struct
import os
import zstandard as zstd
import chess
from typing import Iterator

# ── HalfKP constants ──────────────────────────────────────────────────────────

HALFKP_FEATURES = 64 * 640  # 40,960

# Piece type to index (excluding king): Pawn=0, Knight=1, Bishop=2, Rook=3, Queen=4
PIECE_TYPE_INDEX = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
}

CP_CLIP = 2000  # Centipawn values outside +-2000 are clipped


# ── HalfKP feature extraction ─────────────────────────────────────────────────


def orient_square(sq: int, mirror: bool) -> int:
    """Mirrors a square vertically (for black king POV)."""
    return sq ^ 56 if mirror else sq


def halfkp_features(board: chess.Board) -> tuple[list[int], list[int]]:
    """
    Computes HalfKP feature indices for both sides.

    Returns: (features_white_pov, features_black_pov)
    Both lists contain uint16 indices in [0, 40960).
    """
    white_king_sq = board.king(chess.WHITE)
    black_king_sq = board.king(chess.BLACK)

    white_king_oriented = orient_square(white_king_sq, mirror=False)
    black_king_oriented = orient_square(black_king_sq, mirror=True)

    features_a = []  # White king POV
    features_b = []  # Black king POV

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None or piece.piece_type == chess.KING:
            continue

        pt_idx = PIECE_TYPE_INDEX[piece.piece_type]

        # Color bit: own pieces = 0, opponent pieces = 1 (relative to each king's side)
        color_a = 0 if piece.color == chess.WHITE else 1
        color_b = 0 if piece.color == chess.BLACK else 1

        piece_idx_a = pt_idx * 2 + color_a
        piece_idx_b = pt_idx * 2 + color_b

        sq_oriented_a = orient_square(sq, mirror=False)
        sq_oriented_b = orient_square(sq, mirror=True)

        idx_a = white_king_oriented * 640 + piece_idx_a * 64 + sq_oriented_a
        idx_b = black_king_oriented * 640 + piece_idx_b * 64 + sq_oriented_b

        features_a.append(idx_a)
        features_b.append(idx_b)

    print(features_a, features_b)
    return features_a, features_b


# ── Lichess eval parser ───────────────────────────────────────────────────────


def best_cp(evals: list) -> int | None:
    """
    Selects the best cp score from the eval list.
    Strategy: highest depth, first PV, cp only (no mate scores).
    Returns None if no cp is available (e.g. forced mate).
    """
    best = None
    best_depth = -1

    for ev in evals:
        depth = ev.get("depth", 0)
        pvs = ev.get("pvs", [])
        if not pvs:
            continue
        first_pv = pvs[0]
        if "cp" not in first_pv:
            continue
        if depth > best_depth:
            best_depth = depth
            best = first_pv["cp"]

    return best


def stream_positions(path: str) -> Iterator[tuple[chess.Board, int]]:
    """Streams (board, cp) pairs from the .jsonl.zst file."""
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            buffer = b""
            while True:
                chunk = reader.read(65536)
                if not chunk:
                    if buffer.strip():
                        yield from _parse_lines([buffer])
                    break
                buffer += chunk
                lines = buffer.split(b"\n")
                buffer = lines[-1]
                yield from _parse_lines(lines[:-1])


def _parse_lines(lines: list[bytes]) -> Iterator[tuple[chess.Board, int]]:
    """Parses a list of raw JSON lines into (board, cp) pairs."""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        cp = best_cp(obj.get("evals", []))
        if cp is None:
            continue

        try:
            board = chess.Board(obj["fen"])
        except (ValueError, KeyError):
            continue

        # cp is always stored from white's perspective in Lichess data;
        # flip it so it's always from the perspective of the side to move
        if board.turn == chess.BLACK:
            cp = -cp

        yield board, cp


# ── Binary writer ─────────────────────────────────────────────────────────────


def write_sample(
    out_fh,
    index_fh,
    cp: int,
    features_a: list[int],
    features_b: list[int],
) -> None:
    """Writes one sample to the binary file and its byte offset to the index file."""
    cp_clipped = max(-CP_CLIP, min(CP_CLIP, cp))
    n_a = len(features_a)
    n_b = len(features_b)

    offset = out_fh.tell()
    index_fh.write(struct.pack("<Q", offset))  # uint64 byte offset

    # Header: cp (int16) + n_a (uint8) + n_b (uint8)
    out_fh.write(struct.pack("<hBB", cp_clipped, n_a, n_b))
    # Feature indices
    out_fh.write(struct.pack(f"<{n_a}H", *features_a))
    out_fh.write(struct.pack(f"<{n_b}H", *features_b))


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Lichess eval DB -> HalfKP binary")
    parser.add_argument(
        "--input", required=True, help="Path to lichess_db_eval.jsonl.zst"
    )
    parser.add_argument("--output", required=True, help="Output .bin file")
    parser.add_argument(
        "--limit", type=int, default=None, help="Max number of samples to write"
    )
    parser.add_argument(
        "--skip", type=int, default=0, help="Skip the first N positions"
    )
    args = parser.parse_args()

    index_path = args.output.replace(".bin", ".idx")

    count = 0
    skipped = 0
    errors = 0

    print(f"Input:  {args.input}")
    print(f"Output: {args.output} + {index_path}")
    if args.limit:
        print(f"Limit:  {args.limit:,} samples")

    with open(args.output, "wb") as out_fh, open(index_path, "wb") as idx_fh:
        # File header: magic bytes + version
        out_fh.write(b"NNUE")
        out_fh.write(struct.pack("<I", 1))  # version 1

        for board, cp in stream_positions(args.input):
            if skipped < args.skip:
                skipped += 1
                continue

            try:
                features_a, features_b = halfkp_features(board)
            except Exception:
                errors += 1
                continue

            if not features_a or not features_b:
                continue  # skip empty boards or other edge cases

            write_sample(out_fh, idx_fh, cp, features_a, features_b)
            count += 1

            if count % 100_000 == 0:
                print(f"  {count:>10,} samples  (errors: {errors})", end="\r")

            if args.limit and count >= args.limit:
                break

    print(f"\nDone: {count:,} samples written, {errors} errors skipped.")
    print(
        f"Output: {args.output} ({os.path.getsize(args.output) / 1024 / 1024:.1f} MB)"
    )


if __name__ == "__main__":
    main()
