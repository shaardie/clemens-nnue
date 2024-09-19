import logging
import argparse

import json
import io
import torch
import torch.utils
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
# torch._logging.set_logs(dynamo = logging.DEBUG)
# torch._dynamo.config.verbose = True
# torch._inductor.config.debug = True

logger = logging.getLogger(__name__)

NUM_FEATURES = 64 * 64 * 5 * 2 * 2
M = 4
N = 8
K = 1


class LSB:
    def __iter__(self, bitboard: int):
        self.b = bitboard
        return self

    def __next__(self):
        x = self.bitboard & -self.bitboard
        self.bitboard &= self.bitboard - 1
        return x


PAWN = 0
KNIGHT = 1
BISHOP = 2
ROOK = 3
QUEEN = 4
KING = 5

ExtPieceToClemens = [
    KNIGHT,
    BISHOP,
    ROOK,
    QUEEN,
    KING,
    PAWN,
    ROOK,
    PAWN,
]


WHITE = 0
BLACK = 1


class IterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        with open(self.filename, "rb") as f:
            while True:
                occ_bytes = f.read(8)
                occ = int.from_bytes(occ_bytes, signed=False)
                number_of_pieces = occ.bit_count()
                assert number_of_pieces <= 32
                turn_and_rules50 = int.from_bytes(f.read(1))
                turn = turn_and_rules50 & 1
                assert turn <= 1
                rules50 = turn_and_rules50 >> 1
                assert rules50 <= 100
                packed_pieces_size = (number_of_pieces + 1) // 2
                assert packed_pieces_size <= 16
                packed_pieces = f.read(packed_pieces_size)
                pieces = []
                kings = [None, None]
                for b in packed_pieces:
                    for piece in b & 0x0F, b >> 4:
                        square = (occ & -occ).bit_length() - 1
                        occ &= occ - 1
                        piece_type = ExtPieceToClemens[(piece & 0xFE) // 2]
                        piece_color = piece & 1
                        if piece_type == KING:
                            kings[piece_color] = square
                            continue
                        pieces.append((square, piece_type, piece_color))

                print(pieces)
                return
                # This is an upstream bug
                # https://github.com/lucasart/c-chess-cli/issues/63
                score = int.from_bytes(f.read(4), signed=True, byteorder="little")
                result = int.from_bytes(f.read(4), signed=False, byteorder="little") / 2

                white_features = torch.zeros(NUM_FEATURES)
                black_features = torch.zeros(NUM_FEATURES)

                for piece in pieces:
                    white_features[calc_index(piece, kings[WHITE])] = 1
                    black_features[calc_index(piece, kings[BLACK])] = 1

                white_features = white_features.to_sparse()
                black_features = black_features.to_sparse()
                yield (white_features, black_features, turn, score, result)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, filename):
        super(Dataset).__init__()

        self.filename = filename

        logger.debug(f"Read the file {self.filename} to memory")
        # Read file into
        with open(self.filename, "rb") as f:
            self.data = f.read()

        logger.info(f"examine dataset {self.filename}")
        self.idxs = []
        self.f = io.BytesIO(self.data)
        idx = 0
        while True:
            self.idxs.append(idx)
            occ_bytes = self.f.read(8)
            if len(occ_bytes) == 0:
                break
            number_of_piece_bytes = (
                int.from_bytes(occ_bytes, signed=False).bit_count() + 1
            ) // 2
            idx += 8 + 1 + number_of_piece_bytes + 4 + 4
            self.f.seek(idx)
        self.f.seek(0)
        logger.info(f"found {len(self.idxs)} positions")
        logger.info(f"approx. {self.idxs[-1]} bytes")

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        self.f.seek(self.idxs[idx])
        occ_bytes = self.f.read(8)
        occ = int.from_bytes(occ_bytes, signed=False)
        number_of_pieces = occ.bit_count()
        assert number_of_pieces <= 32
        turn_and_rules50 = int.from_bytes(self.f.read(1))
        turn = turn_and_rules50 & 1
        assert turn <= 1
        rules50 = turn_and_rules50 >> 1
        assert rules50 <= 100
        packed_pieces_size = (number_of_pieces + 1) // 2
        assert packed_pieces_size <= 16
        packed_pieces = self.f.read(packed_pieces_size)
        pieces = []
        kings = [None, None]
        for b in packed_pieces:
            for piece in b & 0x0F, b >> 4:
                square = (occ & -occ).bit_length() - 1
                occ &= occ - 1
                piece_type = ExtPieceToClemens[(piece & 0xFE) // 2]
                piece_color = piece & 1
                if piece_type == KING:
                    kings[piece_color] = square
                    continue
                pieces.append((square, piece_type, piece_color))

        score = int.from_bytes(self.f.read(4), signed=True, byteorder="little")
        result = int.from_bytes(self.f.read(4), signed=False, byteorder="little") / 2

        white_features = torch.zeros(NUM_FEATURES)
        black_features = torch.zeros(NUM_FEATURES)

        for piece in pieces:
            white_features[calc_index(piece, kings[WHITE])] = 1
            black_features[calc_index(piece, kings[BLACK])] = 1

        white_features = white_features.to_sparse()
        black_features = black_features.to_sparse()
        return (white_features, black_features, turn, score, result)


def calc_index(piece, king):
    piece_index = piece[1] * 2 + piece[2]
    return piece[0] + (piece_index + king * 10) * 64


def collate(data):
    data = zip(*data)
    white_features = torch.stack(next(data))
    black_features = torch.stack(next(data))
    turn = torch.tensor(next(data)).reshape(-1, 1)
    score = torch.tensor(next(data)).reshape(-1, 1)
    result = torch.tensor(next(data)).reshape(-1, 1)
    return (
        white_features,
        black_features,
        turn,
        score,
        result,
    )


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

    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"use cuda with GPU {torch.cuda.get_device_name()}")
    else:
        device = "cpu"
        logger.info("use cpu")

    model = NNUE(args.lr, args.lambda_).to(device)

    if args.load_state:
        logger.info(f"load previous model {args.load_state}")
        model.load_state_dict(torch.load(args.load_state, weights_only=True))

    epoch = args.epoch
    logger.info(f"train for {epoch} epochs")

    batch_number = 0
    while epoch > 0:
        logger.info(f"epoch: {epoch}")
        dataset = IterableDataset(args.dataset)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            collate_fn=collate,
        )

        for batch in iter(dataloader):
            model.training_step(batch, batch_number)
            batch_number += 1
            if batch_number % 10 == 0:
                logger.debug(f"trained {batch_number} batches")

                torch.save(model.state_dict(), args.save_state)
                logger.debug(f"saved state to {args.save_state}")

        epoch -= 1

    logger.info("training finished")

    torch.save(model.state_dict(), args.save_state)
    logger.debug(f"saved state to {args.save_state}")

    model.save(args.output_json)
    logger.info(f"stored model in {args.output_json}")


if __name__ == "__main__":
    main()
