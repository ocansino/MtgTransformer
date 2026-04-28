# test_model_pipeline.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import (
    load_train_and_test_sequences,
    build_vocab,
    encode_sequences,
    DraftDataset,
)
from model import DraftTransformer


def main():
    TRAIN_CSV = "train.csv"
    TEST_CSV = "test.csv"
    RATING_TSV = "m19_rating.tsv"
    LAND_RATING_TSV = "m19_land_rating.tsv"

    print("Loading sequences...")
    train_sequences, test_sequences, _ = load_train_and_test_sequences(
        TRAIN_CSV,
        TEST_CSV,
        RATING_TSV,
        LAND_RATING_TSV,
        expected_len=45,
        verbose=False,
    )

    print("Building vocab...")
    vocab = build_vocab(train_sequences)
    print(f"Vocab size: {vocab.size}")

    print("Encoding sequences...")
    train_encoded = encode_sequences(train_sequences, vocab)
    test_encoded = encode_sequences(test_sequences, vocab)

    print("Building datasets...")
    train_dataset = DraftDataset(train_encoded, seq_len=45)
    test_dataset = DraftDataset(test_encoded, seq_len=45)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    print("Creating dataloader...")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    print("Building model...")
    model = DraftTransformer(
        vocab_size=vocab.size,
        seq_len=44,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=512,
        dropout=0.1,
        pad_id=vocab.pad_id,
    )

    # Get one real batch
    x, y = next(iter(train_loader))

    print("Batch input shape:", x.shape)
    print("Batch target shape:", y.shape)

    # Forward pass
    logits = model(x)

    print("Logits shape:", logits.shape)

    # Cross-entropy loss expects:
    # logits: (batch_size * seq_len, vocab_size)
    # targets: (batch_size * seq_len)
    logits_flat = logits.reshape(-1, vocab.size)
    y_flat = y.reshape(-1)

    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits_flat, y_flat)

    print("Flattened logits shape:", logits_flat.shape)
    print("Flattened targets shape:", y_flat.shape)
    print("Loss:", loss.item())


if __name__ == "__main__":
    main()