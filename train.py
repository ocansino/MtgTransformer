# train.py

from __future__ import annotations
from torch.utils.data import Subset
import os
import random
import time
import pandas as pd
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data import (
    load_train_and_test_sequences,
    build_vocab,
    encode_sequences,
    DraftDataset,
)
from model import DraftTransformer


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_top1_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    logits: (batch_size, seq_len, vocab_size)
    targets: (batch_size, seq_len)
    """
    preds = torch.argmax(logits, dim=-1)  # (batch_size, seq_len)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / total

def compute_topk_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k: int = 5,
) -> float:
    """
    logits: (batch_size, seq_len, vocab_size)
    targets: (batch_size, seq_len)

    Returns:
        Fraction of positions where the true target is in the model's top-k predictions.
    """
    topk = torch.topk(logits, k=k, dim=-1).indices  # (batch_size, seq_len, k)
    targets_expanded = targets.unsqueeze(-1)        # (batch_size, seq_len, 1)

    correct = (topk == targets_expanded).any(dim=-1).sum().item()
    total = targets.numel()
    return correct / total


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()

    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    num_batches = 0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)  # (batch_size, seq_len, vocab_size)

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )

        loss.backward()
        optimizer.step()

        top1 = compute_top1_accuracy(logits, y)
        top5 = compute_topk_accuracy(logits, y, k=5)

        total_loss += loss.item()
        total_top1 += top1
        total_top5 += top5
        num_batches += 1

    avg_loss = total_loss / num_batches
    avg_top1 = total_top1 / num_batches
    avg_top5 = total_top5 / num_batches
    return avg_loss, avg_top1, avg_top5


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()

    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    num_batches = 0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )

        top1 = compute_top1_accuracy(logits, y)
        top5 = compute_topk_accuracy(logits, y, k=5)

        total_loss += loss.item()
        total_top1 += top1
        total_top5 += top5
        num_batches += 1

    avg_loss = total_loss / num_batches
    avg_top1 = total_top1 / num_batches
    avg_top5 = total_top5 / num_batches
    return avg_loss, avg_top1, avg_top5

@torch.no_grad()
def evaluate_by_position(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    seq_len: int = 44,
) -> tuple[list[float], list[float]]:
    """
    Computes top-1 and top-5 accuracy separately for each prediction position.

    Position 0 predicts original draft card 1.
    Position 43 predicts original draft card 44.
    """
    model.eval()

    top1_correct = torch.zeros(seq_len, device=device)
    top5_correct = torch.zeros(seq_len, device=device)
    totals = torch.zeros(seq_len, device=device)

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)

        top1_preds = torch.argmax(logits, dim=-1)
        top5_preds = torch.topk(logits, k=5, dim=-1).indices

        top1_matches = top1_preds == y
        top5_matches = (top5_preds == y.unsqueeze(-1)).any(dim=-1)

        top1_correct += top1_matches.sum(dim=0)
        top5_correct += top5_matches.sum(dim=0)
        totals += torch.tensor(y.shape[0], device=device).repeat(seq_len)

    top1_by_pos = (top1_correct / totals).cpu().tolist()
    top5_by_pos = (top5_correct / totals).cpu().tolist()

    return top1_by_pos, top5_by_pos


def main():
    # ----------------------------
    # Config
    # ----------------------------
    start_time = time.time()
    TRAIN_CSV = "train.csv"
    TEST_CSV = "test.csv"
    RATING_TSV = "m19_rating.tsv"
    LAND_RATING_TSV = "m19_land_rating.tsv"

    RESUME_FROM_CHECKPOINT = False
    CHECKPOINT_DIR = "checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    LAST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "last_checkpoint.pt")
    BEST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")

    BATCH_SIZE = 64
    D_MODEL = 128
    NHEAD = 4
    NUM_LAYERS = 2
    DIM_FEEDFORWARD = 512
    DROPOUT = 0.1
    LEARNING_RATE = 3e-4
    EPOCHS = 3
    VAL_SPLIT = 0.1
    SEED = 42

    

    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ----------------------------
    # Load and preprocess data
    # ----------------------------
    print("Loading sequences...")
    train_sequences, test_sequences, _ = load_train_and_test_sequences(
        TRAIN_CSV,
        TEST_CSV,
        RATING_TSV,
        LAND_RATING_TSV,
        expected_len=45,
        verbose=False,
    )

    print("Building vocabulary...")
    vocab = build_vocab(train_sequences)
    print(f"Vocab size: {vocab.size}")

    print("Encoding sequences...")
    train_encoded = encode_sequences(train_sequences, vocab)
    test_encoded = encode_sequences(test_sequences, vocab)

    print("Building datasets...")
    full_train_dataset = DraftDataset(train_encoded, seq_len=45)
    test_dataset = DraftDataset(test_encoded, seq_len=45)

    print(f"Full train dataset size: {len(full_train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    USE_SUBSET = True
    TRAIN_SUBSET_SIZE = 300000
    TEST_SUBSET_SIZE = 60000

    if USE_SUBSET:
        print("\nUsing subset of data for faster training...")

        full_train_dataset = Subset(
            full_train_dataset,
            range(min(TRAIN_SUBSET_SIZE, len(full_train_dataset)))
        )

        test_dataset = Subset(
            test_dataset,
            range(min(TEST_SUBSET_SIZE, len(test_dataset)))
        )

        print(f"Subset train size: {len(full_train_dataset)}")
        print(f"Subset test size: {len(test_dataset)}")

    # ----------------------------
    # Train / validation split
    # ----------------------------
    val_size = int(len(full_train_dataset) * VAL_SPLIT)
    train_size = len(full_train_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    print(f"Train split size: {len(train_dataset)}")
    print(f"Validation split size: {len(val_dataset)}")

    # ----------------------------
    # Dataloaders
    # ----------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # ----------------------------
    # Model, optimizer, loss
    # ----------------------------
    model = DraftTransformer(
        vocab_size=vocab.size,
        seq_len=44,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        pad_id=vocab.pad_id,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 1
    best_val_loss = float("inf")
    best_checkpoint_path = BEST_CHECKPOINT_PATH
    last_checkpoint_path = LAST_CHECKPOINT_PATH

    if RESUME_FROM_CHECKPOINT and os.path.exists(last_checkpoint_path):
        print(f"Resuming training from {last_checkpoint_path}")
        checkpoint = torch.load(last_checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]

        print(f"Resumed from epoch {checkpoint['epoch']}")
        print(f"Best validation loss so far: {best_val_loss:.4f}")

    # ----------------------------
    # Training loop
    # ----------------------------
    

    for epoch in range(start_epoch, EPOCHS + 1):
        train_loss, train_top1, train_top5 = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_loss, val_top1, val_top5 = evaluate(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train Top-1: {train_top1:.4f} | Train Top-5: {train_top5:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Top-1: {val_top1:.4f} | Val Top-5: {val_top5:.4f}"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "vocab_size": vocab.size,
                "pad_id": vocab.pad_id,
                "config": {
                    "seq_len": 44,
                    "d_model": D_MODEL,
                    "nhead": NHEAD,
                    "num_layers": NUM_LAYERS,
                    "dim_feedforward": DIM_FEEDFORWARD,
                    "dropout": DROPOUT,
                },
            },
            last_checkpoint_path,
        )
        print(f"Saved latest checkpoint to {last_checkpoint_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab_size": vocab.size,
                    "pad_id": vocab.pad_id,
                    "config": {
                        "seq_len": 44,
                        "d_model": D_MODEL,
                        "nhead": NHEAD,
                        "num_layers": NUM_LAYERS,
                        "dim_feedforward": DIM_FEEDFORWARD,
                        "dropout": DROPOUT,
                    },
                },
                best_checkpoint_path,
            )
            print(f"Saved new best model to {best_checkpoint_path}") 
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "vocab_size": vocab.size,
                "pad_id": vocab.pad_id,
                "config": {
                    "seq_len": 44,
                    "d_model": D_MODEL,
                    "nhead": NHEAD,
                    "num_layers": NUM_LAYERS,
                    "dim_feedforward": DIM_FEEDFORWARD,
                    "dropout": DROPOUT,
                },
            },
            last_checkpoint_path,
        )
        print(f"Saved latest checkpoint to {last_checkpoint_path}")    

    # ----------------------------
    # Final test evaluation
    # ----------------------------
    print("\nLoading best checkpoint for final test evaluation...")
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_top1, test_top5 = evaluate(
        model, test_loader, criterion, device
    )

    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Final Test Top-1 Accuracy: {test_top1:.4f}")
    print(f"Final Test Top-5 Accuracy: {test_top5:.4f}")

    print("\nEvaluating accuracy by pick position...")
    top1_by_pos, top5_by_pos = evaluate_by_position(
        model,
        test_loader,
        device,
        seq_len=44,
    )

    print("\nAccuracy by prediction position:")
    print("Position | Top-1 | Top-5")
    print("------------------------")
    for pos, (top1, top5) in enumerate(zip(top1_by_pos, top5_by_pos), start=1):
        print(f"{pos:8d} | {top1:.4f} | {top5:.4f}")

    position_results = pd.DataFrame({
        "position": list(range(1, 45)),
        "top1_accuracy": top1_by_pos,
        "top5_accuracy": top5_by_pos,
    })

    position_results.to_csv("accuracy_by_position.csv", index=False)
    print("\nSaved position accuracy results to accuracy_by_position.csv")
    end_time = time.time()
    print(f"\nTotal runtime: {(end_time - start_time)/60:.2f} minutes")


if __name__ == "__main__":
    main()