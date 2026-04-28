# train.py

from __future__ import annotations
from torch.utils.data import Subset
import os
import random
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


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()

    total_loss = 0.0
    total_acc = 0.0
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

        acc = compute_top1_accuracy(logits, y)

        total_loss += loss.item()
        total_acc += acc
        num_batches += 1

    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()

    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )

        acc = compute_top1_accuracy(logits, y)

        total_loss += loss.item()
        total_acc += acc
        num_batches += 1

    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    return avg_loss, avg_acc


def main():
    # ----------------------------
    # Config
    # ----------------------------
    TRAIN_CSV = "train.csv"
    TEST_CSV = "test.csv"
    RATING_TSV = "m19_rating.tsv"
    LAND_RATING_TSV = "m19_land_rating.tsv"

    BATCH_SIZE = 64
    D_MODEL = 128
    NHEAD = 4
    NUM_LAYERS = 2
    DIM_FEEDFORWARD = 512
    DROPOUT = 0.1
    LEARNING_RATE = 3e-4
    EPOCHS = 1
    VAL_SPLIT = 0.1
    SEED = 42

    CHECKPOINT_DIR = "checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

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
    TRAIN_SUBSET_SIZE = 50000
    TEST_SUBSET_SIZE = 10000

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

    # ----------------------------
    # Training loop
    # ----------------------------
    best_val_loss = float("inf")
    best_checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

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

    # ----------------------------
    # Final test evaluation
    # ----------------------------
    print("\nLoading best checkpoint for final test evaluation...")
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc = evaluate(
        model, test_loader, criterion, device
    )

    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Final Test Top-1 Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()