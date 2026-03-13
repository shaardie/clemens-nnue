#!/usr/bin/env python3
"""
Re-export an existing trained model with different quantization settings.

Usage:
    python export.py <pytorch_model.pt> <output.bin>
    python export.py model.pt nnue.bin
"""

import argparse
import struct
import numpy
import torch
from train import NNUEModel, INPUT_SIZE, HIDDEN_SIZE, L1_SIZE, L2_SIZE
from train import MODEL_FILE as INPUT_FILE

OUTPUT_FILE = f"nnue-h{HIDDEN_SIZE}-l{L1_SIZE}.bin"
QUANT_SCALE = 256


def save_weights(model: NNUEModel, path: str):
    """Export weights with int16-quantized feature transformer."""
    with open(path, "wb") as f:
        for v in (INPUT_SIZE, HIDDEN_SIZE, L1_SIZE, L2_SIZE):
            f.write(struct.pack("<I", v))

        def write_f32(t):
            arr = t.detach().cpu().numpy().astype(numpy.float32)
            f.write(arr.tobytes())

        def write_i16(t):
            arr = t.detach().cpu().numpy()
            arr = numpy.clip(numpy.round(arr * QUANT_SCALE), -32768, 32767)
            f.write(arr.astype(numpy.int16).tobytes())

        write_i16(model.ft.weight.t())
        write_i16(model.ft.bias)

        write_f32(model.l1.weight)
        write_f32(model.l1.bias)
        write_f32(model.l2.weight)
        write_f32(model.l2.bias)
        write_f32(model.out.weight)
        write_f32(model.out.bias)


if __name__ == "__main__":
    model = NNUEModel()
    model.load_state_dict(torch.load(INPUT_FILE, map_location="cpu"))
    model.eval()
    save_weights(model, OUTPUT_FILE)
    print(f"Exported {INPUT_FILE} → {OUTPUT_FILE}")
