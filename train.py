import logging
import argparse

import json
import ctypes
import torch
import torch.utils
import torch.utils.data
import numpy as np  # silly, but easier to use other code
from torch.utils.tensorboard.writer import SummaryWriter

MAX_ACTIVE_FEATURES = 32

# External C Library
libdataset = ctypes.CDLL("./build/libdataset.so")

CreateBatchStream = libdataset.CreateBatchStream
CreateBatchStream.argtypes = [ctypes.c_char_p, ctypes.c_uint]
CreateBatchStream.restype = ctypes.c_void_p

DestroyBatchStream = libdataset.DestroyBatchStream
DestroyBatchStream.argtypes = [ctypes.c_void_p]
DestroyBatchStream.restype = None


class SparseBatch(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_int),
        ("num_active_features", ctypes.c_int),
        ("stm", ctypes.POINTER(ctypes.c_int)),
        ("score", ctypes.POINTER(ctypes.c_int)),
        ("result", ctypes.POINTER(ctypes.c_float)),
        ("white_features_indices", ctypes.POINTER(ctypes.c_int)),
        ("black_features_indices", ctypes.POINTER(ctypes.c_int)),
    ]

    def get_tensors(self):
        stm = torch.from_numpy(np.ctypeslib.as_array(self.score, shape=(self.size, 1)))
        score = torch.from_numpy(
            np.ctypeslib.as_array(self.score, shape=(self.size, 1))
        )
        result = torch.from_numpy(
            np.ctypeslib.as_array(self.result, shape=(self.size, 1))
        )

        # As we said, the index tensor needs to be transposed (not the whole sparse tensor!).
        # This is just how pytorch stores indices in sparse tensors.
        # It also requires the indices to be 64-bit ints.
        white_features_indices = torch.transpose(
            torch.from_numpy(
                np.ctypeslib.as_array(
                    self.white_features_indices,
                    shape=(self.num_active_features, 2),
                )
            ),
            0,
            1,
        ).long()
        black_features_indices = torch.transpose(
            torch.from_numpy(
                np.ctypeslib.as_array(
                    self.black_features_indices,
                    shape=(self.num_active_features, 2),
                )
            ),
            0,
            1,
        ).long()

        # The values are all ones, so we can create these tensors in place easily.
        # No need to go through a copy.
        white_features_values = torch.ones(self.num_active_features)
        black_features_values = torch.ones(self.num_active_features)

        # Now the magic. We construct a sparse tensor by giving the indices of
        # non-zero values (active feature indices) and the values themselves (all ones!).
        # The size of the tensor is batch_size*NUM_FEATURES, which would
        # normally be insanely large, but since the density is ~0.1% it takes
        # very little space and allows for faster forward pass.
        # For maximum performance we do cheat somewhat though. Normally pytorch
        # checks the correctness, which is an expensive O(n) operation.
        # By using _sparse_coo_tensor_unsafe we avoid that.
        white_features = torch.sparse_coo_tensor(
            white_features_indices,
            white_features_values,
            (self.size, NUM_FEATURES),
            is_coalesced=True,
        )
        black_features = torch.sparse_coo_tensor(
            black_features_indices,
            black_features_values,
            (self.size, NUM_FEATURES),
            is_coalesced=True,
        )

        # What is coalescing?! It makes sure the indices are unique and ordered.
        # Now you probably see why we said the inputs must be ordered from the start.
        # This is normally a O(n log n) operation and takes a significant amount of
        # time. But here we **know** that the tensor is already in a coalesced form,
        # therefore we can just tell pytorch that it can use that assumption.
        white_features._coalesced_(True)
        black_features._coalesced_(True)

        return (white_features, black_features, stm, score, result)


SparseBatchPtr = ctypes.POINTER(SparseBatch)

GetNextBatch = libdataset.GetNextBatch
GetNextBatch.argtypes = [ctypes.c_void_p]
GetNextBatch.restype = SparseBatchPtr

DestroyBatch = libdataset.DestroyBatch
DestroyBatch.argtypes = [SparseBatchPtr]
DestroyBatch.restype = None


writer = SummaryWriter()
# torch._logging.set_logs(dynamo = logging.DEBUG)
# torch._dynamo.config.verbose = True
# torch._inductor.config.debug = True

logger = logging.getLogger(__name__)

NUM_FEATURES = 64 * 64 * 5 * 2 * 2
M = 4
N = 8
K = 1


class NNUE(torch.nn.Module):
    def __init__(self, lr, lambda_):
        super(NNUE, self).__init__()

        self.l0 = torch.nn.Linear(NUM_FEATURES, M)
        self.l1 = torch.nn.Linear(2 * M, N)
        self.l2 = torch.nn.Linear(N, K)

        self.lambda_ = lambda_

        self.optimizer = torch.optim
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)

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

    def training_step(self, batch, batch_number):
        # Zero your gradients for every batch!
        self.optimizer.zero_grad()

        # Make predictions for this batch
        output = self(*batch)

        writer.add_histogram("output/train", output, batch_number)

        writer.add_scalars(
            "output/train",
            {
                "max": output.max(),
                "min": output.min(),
                "mean": output.mean(),
            },
            batch_number,
        )

        # Compute the loss and its gradients
        loss = self.loss(batch, output)
        writer.add_scalar("Loss/train", loss, batch_number)

        # Adjust learning weights
        loss.backward()
        self.optimizer.step()

    def loss(self, batch, output):
        white_features, black_features, turn, score, result = batch

        # Loss function
        scaling_factor = 10  # TODO better value
        lambda_ = self.lambda_
        wdl_eval_model = torch.sigmoid(output / scaling_factor)
        wdl_eval_target = torch.sigmoid(score / scaling_factor)
        wdl_value_target = lambda_ * wdl_eval_target + (1 - lambda_) * result
        loss = torch.pow(wdl_eval_model - wdl_value_target, 2)
        return loss.mean()

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


def init():
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="be more verbose",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="datasets used for training, can be given multiple times",
    )
    parser.add_argument(
        "--output-json",
        default="model.json",
        help="output file in json, defaults to model.json",
    )

    parser.add_argument(
        "--save-state",
        default="model.pt",
        help="save the state of the model in a .pt file",
    )

    parser.add_argument("--load-state", help="path to starting model starting model")

    parser.add_argument("--epoch", type=int, default=1, help="epoch, defaults 1")
    parser.add_argument(
        "--batch-size", type=int, default=8192, help="batch size, defaults 8192"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="learning rate, defaults to 20, which is silly",
    )
    parser.add_argument(
        "--lambda", dest="lambda_", type=float, default=0.5, help="lambda"
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    return args


def main():
    args = init()

    # Check if CUDA is available and select the GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        print(
            f"CUDA is available! You have {torch.cuda.device_count()} CUDA-capable devices."
        )
    else:
        print("CUDA is not available. The GPU will not be used.")

    model = NNUE(args.lr, args.lambda_).to(device)

    if args.load_state:
        logger.info(f"load previous model {args.load_state}")
        model.load_state_dict(torch.load(args.load_state, weights_only=True))

    epoch = args.epoch
    logger.info(f"train for {epoch} epochs")

    batch_number = 0
    while epoch > 0:
        batchstream = CreateBatchStream(args.dataset.encode("utf-8"), args.batch_size)
        logger.info(f"epoch: {epoch}")
        while True:
            sparseBatchPtr = GetNextBatch(batchstream)
            try:
                batch = sparseBatchPtr.contents.get_tensors()
            except ValueError as e:
                logger.info("NULL Pointer, so probably end of file: %s", e)
                break
            model.training_step(batch, batch_number)
            batch_number += 1
            if batch_number % 1000 == 0:
                torch.save(model.state_dict(), args.save_state)
                logger.debug(f"saved state to {args.save_state}")
            DestroyBatch(sparseBatchPtr)
        DestroyBatchStream(batchstream)
        epoch -= 1

    logger.info("training finished")

    torch.save(model.state_dict(), args.save_state)
    logger.debug(f"saved state to {args.save_state}")

    model.save(args.output_json)
    logger.info(f"stored model in {args.output_json}")


if __name__ == "__main__":
    main()
