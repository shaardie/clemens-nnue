#!/usr/bin/env python3

from typing import Iterator, Optional, Tuple

import zstandard
import sys
import csv
import json

# Etract 5 000 000 positions
NUMBER_OF_POSITIONS = 5000000
OUTPUT = "eval.csv"


def read_zsdt_by_line(path: str) -> Iterator[bytes]:
    """
    Read the zst compressed lichess database line by line
    """
    dctx = zstandard.ZstdDecompressor()
    with open(path, "rb") as f:
        with dctx.stream_reader(f) as reader:
            buffer = b""
            while True:
                chunk = reader.read(1 << 16)
                if not chunk:
                    break  # there is still something, but we do not care
                buffer += chunk
                lines = buffer.split(b"\n")
                # completed lines are all but the last one and these we can yield
                # the last one is left for further completion
                buffer = lines[-1]
                lines = lines[:-1]
                for line in lines:
                    yield line


def parse_line(line: bytes) -> Optional[Tuple[str, int]]:
    """
    Parse a line from the lichess database.
    It is in json including a fen.

    Note from the lichess database: Evaluations have various depths and node
    count. If you only want one PV, we recommend selecting the evaluation with
    the highest depth, and use its first PV.

    Returns fen and evaluation in centipawns on success and None otherwise
    """
    obj = json.loads(line)
    # get
    fen = obj["fen"]
    evals = obj.get("evals", [])
    best_eval = evals[0]
    best_depth = best_eval["depth"]
    for current_eval in evals[1:]:
        current_depth = current_eval["depth"]
        if current_depth < best_depth:
            continue
        best_eval = current_eval
        best_depth = current_depth
    cp = best_eval["pvs"][0].get("cp", None)
    if cp is None:
        return
    return fen, cp


if __name__ == "__main__":
    number_of_positions = 0
    with open(OUTPUT, "w") as f:
        csv_writer = csv.writer(f)
        for x in read_zsdt_by_line(sys.argv[1]):
            r = parse_line(x)
            if r is None:
                continue

            csv_writer.writerow(r)
            number_of_positions += 1
            if number_of_positions % 100_00 == 0:
                print(
                    f"{number_of_positions}/{NUMBER_OF_POSITIONS} completed", end="\r"
                )
            if number_of_positions == NUMBER_OF_POSITIONS:
                break
