#!/usr/bin/env python3

import csv
from typing import List, Optional, Tuple

import chess
import numpy

from lichess_to_csv import OUTPUT as INPUT
from lichess_to_csv import NUMBER_OF_POSITIONS

OUTPUT = "features.csv"
CLAMP_SIZE = 2000
MAX_PIECES = 32

PIECE_TYPE_TO_NUMBER = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

PIECE_OFFSET = len(PIECE_TYPE_TO_NUMBER)


def board_to_feature_indices(
    board: chess.Board,
) -> Optional[Tuple[List[int], List[int]]]:
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


if __name__ == "__main__":
    with open(INPUT, "r") as input:
        csv_reader = csv.reader(input)

        output = numpy.full(
            (NUMBER_OF_POSITIONS, MAX_PIECES * 2 + 1), -1, dtype=numpy.int16
        )

        number_of_positions = 0
        for fen, eval in csv_reader:
            board = chess.Board(fen=fen)
            board.pieces_mask()
            indices = board_to_feature_indices(board)
            if indices is None:
                continue
            white_indices, black_indices = indices
            eval = max(min(int(eval), CLAMP_SIZE), -CLAMP_SIZE)
            if board.turn == chess.BLACK:
                eval *= -1
                stm_indices = black_indices
                opp_indices = white_indices
            else:
                stm_indices = white_indices
                opp_indices = black_indices

            output[number_of_positions, 0 : len(stm_indices)] = stm_indices
            output[number_of_positions, MAX_PIECES : MAX_PIECES + len(stm_indices)] = (
                stm_indices
            )
            output[number_of_positions, MAX_PIECES * 2] = eval
            if number_of_positions % 100_00 == 0:
                print(
                    f"{number_of_positions}/{NUMBER_OF_POSITIONS} completed", end="\r"
                )
            number_of_positions += 1

        numpy.savetxt(OUTPUT[:number_of_positions], output, delimiter=",", fmt="%d")
