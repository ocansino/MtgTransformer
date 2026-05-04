# visualize_predictions.py

import torch

from data import (
    load_train_and_test_sequences,
    build_vocab,
    encode_sequences,
    DraftDataset,
    decode_ids,
)
from model import DraftTransformer


def print_prediction_example(
    model,
    sequence_ids,
    vocab,
    position,
    device,
    top_k=5,
):
    """
    Predict card at index `position`.

    The model input is sequence positions 0..43.
    The target at model output position position-1 predicts original sequence[position].
    """
    model.eval()

    input_ids = sequence_ids[:-1]   # length 44
    target_ids = sequence_ids[1:]   # length 44

    x = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)

    output_pos = position - 1
    scores = logits[0, output_pos]
    probs = torch.softmax(scores, dim=-1)

    top_probs, top_ids = torch.topk(probs, k=top_k)

    true_id = target_ids[output_pos]
    true_card = vocab.id_to_token[true_id]

    previous_cards = decode_ids(sequence_ids[:position], vocab)

    print("=" * 80)
    print(f"Predicting pick position {position}")
    print()

    print("Previous picks:")
    for i, card in enumerate(previous_cards, start=1):
        print(f"{i:2d}. {card}")

    print()
    print(f"Actual next card: {true_card}")
    print()
    print(f"Model top-{top_k} predictions:")

    for rank, (card_id, prob) in enumerate(zip(top_ids.tolist(), top_probs.tolist()), start=1):
        card_name = vocab.id_to_token[card_id]
        marker = "✅" if card_name == true_card else ""
        print(f"{rank}. {card_name:30s} prob={prob:.4f} {marker}")

    print()


def main():
    TRAIN_CSV = "train.csv"
    TEST_CSV = "test.csv"
    RATING_TSV = "m19_rating.tsv"
    LAND_RATING_TSV = "m19_land_rating.tsv"
    CHECKPOINT_PATH = "checkpoints/best_model.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data...")
    train_sequences, test_sequences, _ = load_train_and_test_sequences(
        TRAIN_CSV,
        TEST_CSV,
        RATING_TSV,
        LAND_RATING_TSV,
        expected_len=45,
        verbose=False,
    )

    print("Rebuilding vocab...")
    vocab = build_vocab(train_sequences)

    print("Encoding test data...")
    test_encoded = encode_sequences(test_sequences, vocab)

    print("Loading checkpoint...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    config = checkpoint["config"]

    model = DraftTransformer(
        vocab_size=checkpoint["vocab_size"],
        seq_len=config["seq_len"],
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"],
        pad_id=checkpoint["pad_id"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    # Choose a few example sequences and positions.
    # Positions are original draft positions: 1..44 can be predicted.
    examples = [
        (0, 5),
        (0, 20),
        (0, 40),
        (1, 10),
        (2, 30),
    ]

    for seq_index, position in examples:
        sequence_ids = test_encoded[seq_index]
        print_prediction_example(
            model=model,
            sequence_ids=sequence_ids,
            vocab=vocab,
            position=position,
            device=device,
            top_k=5,
        )


if __name__ == "__main__":
    main()