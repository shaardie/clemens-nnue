import torch
import logging
from torch.utils.tensorboard.writer import SummaryWriter
from model import NNUE
import data_loader

writer = SummaryWriter()


def train_model(model: NNUE, config: dict, device: torch.device):
    epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    cache_size = config["training"]["cache_size"]
    dataset_path = config["dataset"]["path"]
    lambda_ = config["training"]["lambda"]
    scaling_factor = config["training"]["scaling_factor"]

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        momentum=config["training"]["momentum"],
    )

    for layer in [model.l0, model.l1, model.l2]:
        if hasattr(layer, "weight"):
            torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
        if hasattr(layer, "bias") and layer.bias is not None:
            torch.nn.init.zeros_(layer.bias)

    batch_number = 0
    logging.info(f"Training for {epochs} epochs...")

    for epoch in range(epochs):
        logging.info(f"Epoch {epoch + 1}/{epochs}")
        batchstream = data_loader.CreateBatchStream(
            dataset_path.encode("utf-8"),
            batch_size,
            cache_size,
        )
        while True:
            sparseBatchPtr = data_loader.GetNextBatch(batchstream)
            try:
                batch = sparseBatchPtr.contents.get_tensors(device)
            except ValueError as e:
                logging.info("NULL Pointer, so probably end of file: %s", e)
                break

            loss = training_step(
                model, batch, optimizer, batch_number, lambda_, scaling_factor
            )

            batch_number += 1
            if batch_number % 1000 == 0:
                logging.debug(f"loss {loss}")
                writer.add_scalar(
                    "Loss/train",
                    loss,
                    batch_number,
                )
            data_loader.DestroyBatch(sparseBatchPtr)
        data_loader.DestroyBatchStream(batchstream)
        # save model after every epoch
        torch.save(model.state_dict(), "model_weights.pth")
        for name, param in model.named_parameters():
            writer.add_scalar(f"Weight_norm/{name}", param.norm().item(), epoch)
            writer.add_histogram(f"Parameters/{name}", param, epoch)


def training_step(model: NNUE, batch, optimizer, batch_number, lambda_, scaling_factor):
    # Zero your gradients for every batch!
    optimizer.zero_grad()

    # Make predictions for this batch
    output = model(*batch)

    # Compute the loss and its gradients
    loss = compute_loss(batch, output, lambda_, batch_number, scaling_factor)

    # Adjust learning weights
    loss.backward()

    # clip Gradient Norm
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    if batch_number % 1000 == 0:
        for name, param in model.named_parameters():
            if param.grad is not None:
                writer.add_histogram(f"Gradients/{name}", param.grad, batch_number)
                writer.add_scalar(
                    f"Gradients/{name}_norm", param.grad.norm().item(), batch_number
                )
    optimizer.step()

    return loss


def compute_loss(batch, output, lambda_, batch_number, scaling_factor):
    white_features, black_features, turn, score, result = batch

    # Loss function
    # wdl_eval_model = torch.sigmoid(output / scaling_factor)
    # wdl_eval_target = torch.sigmoid(score / scaling_factor)
    # wdl_value_target = lambda_ * wdl_eval_target + (1 - lambda_) * result
    if batch_number % 1000 == 0:
        # writer.add_histogram("Model/value", wdl_eval_model, batch_number)
        writer.add_histogram("Model/score", output, batch_number)
        # writer.add_scalars(
        #     "Predictions/output",
        #     {
        #         "max": output.max().item(),
        #         "min": output.min().item(),
        #         "mean": output.mean().item(),
        #     },
        #     batch_number,
        # )
        # writer.add_histogram("Target/value", wdl_value_target, batch_number)
        writer.add_histogram("Target/score", score, batch_number)
    return torch.nn.functional.mse_loss(output, score.to(torch.float32))
    return torch.nn.functional.mse_loss(wdl_eval_model, wdl_value_target)
