# baselines.py
#we are doing baseline tests to compare the transformer with

from collections import Counter
from typing import Dict, List, Tuple

from data import load_train_and_test_sequences


def train_global_frequency_baseline(train_sequences: List[List[str]]) -> Tuple[str, Counter]:
    """
    Train a global frequency baseline using target cards only.

    For each sequence [c1, c2, ..., c45], the autoregressive targets are [c2, ..., c45].
    We count how often each target card appears, then always predict the most common one.

    Returns:
        most_common_card: the single card predicted for every example
        counter: frequency counts of all target cards
    """
    counter = Counter()

    for seq in train_sequences:
        targets = seq[1:]   # autoregressive targets
        counter.update(targets)

    most_common_card, _ = counter.most_common(1)[0]
    return most_common_card, counter


def evaluate_global_frequency_baseline(
    test_sequences: List[List[str]],
    predicted_card: str,
) -> float:
    """
    Evaluate top-1 accuracy of the global frequency baseline.

    Always predicts the same card for every target position.
    """
    correct = 0
    total = 0

    for seq in test_sequences:
        targets = seq[1:]
        for true_card in targets:
            total += 1
            if true_card == predicted_card:
                correct += 1

    return correct / total

def train_position_baseline(train_sequences: List[List[str]]) -> Tuple[Dict[int, str], Dict[int, Counter]]:
    """
    Train a position-conditioned frequency baseline.

    For each target position t in [1..44], count the most common card at that
    exact position in the training set.

    Returns:
        position_to_card: maps target position -> most common card
        position_counters: maps target position -> Counter of cards
    """
    position_counters: Dict[int, Counter] = {}

    for target_pos in range(1, 45):
        position_counters[target_pos] = Counter()

    for seq in train_sequences:
        for target_pos in range(1, 45):
            true_card = seq[target_pos]
            position_counters[target_pos].update([true_card])

    position_to_card = {
        pos: counter.most_common(1)[0][0]
        for pos, counter in position_counters.items()
    }

    return position_to_card, position_counters


def evaluate_position_baseline(
    test_sequences: List[List[str]],
    position_to_card: Dict[int, str],
) -> float:
    """
    Evaluate top-1 accuracy of the position-based baseline.
    """
    correct = 0
    total = 0

    for seq in test_sequences:
        for target_pos in range(1, 45):
            predicted_card = position_to_card[target_pos]
            true_card = seq[target_pos]

            total += 1
            if predicted_card == true_card:
                correct += 1

    return correct / total

def train_bigram_baseline(train_sequences: List[List[str]]) -> Tuple[Dict[str, str], Dict[str, Counter]]:
    """
    Train a bigram baseline:
    predict the next card using only the immediately previous card.

    Returns:
        next_card_map: maps previous_card -> most common next card
        bigram_counters: maps previous_card -> Counter of next cards
    """
    bigram_counters: Dict[str, Counter] = {}

    for seq in train_sequences:
        for i in range(1, len(seq)):
            prev_card = seq[i - 1]
            next_card = seq[i]

            if prev_card not in bigram_counters:
                bigram_counters[prev_card] = Counter()

            bigram_counters[prev_card].update([next_card])

    next_card_map = {
        prev_card: counter.most_common(1)[0][0]
        for prev_card, counter in bigram_counters.items()
    }

    return next_card_map, bigram_counters


def evaluate_bigram_baseline(
    test_sequences: List[List[str]],
    next_card_map: Dict[str, str],
    fallback_card: str,
) -> float:
    """
    Evaluate top-1 accuracy of the bigram baseline.

    If previous card was never seen in training, fall back to the global most common card.
    """
    correct = 0
    total = 0

    for seq in test_sequences:
        for i in range(1, len(seq)):
            prev_card = seq[i - 1]
            true_card = seq[i]

            predicted_card = next_card_map.get(prev_card, fallback_card)

            total += 1
            if predicted_card == true_card:
                correct += 1

    return correct / total

if __name__ == "__main__":
    TRAIN_CSV = "train.csv"
    TEST_CSV = "test.csv"
    RATING_TSV = "m19_rating.tsv"
    LAND_RATING_TSV = "m19_land_rating.tsv"

    train_sequences, test_sequences, _ = load_train_and_test_sequences(
        TRAIN_CSV,
        TEST_CSV,
        RATING_TSV,
        LAND_RATING_TSV,
        expected_len=45,
        verbose=False,
    )

    most_common_card, counter = train_global_frequency_baseline(train_sequences)
    global_accuracy = evaluate_global_frequency_baseline(test_sequences, most_common_card)

    print("=== Global Frequency Baseline ===")
    print(f"Most common predicted card: {most_common_card}")
    print(f"Train frequency of predicted card: {counter[most_common_card]}")
    print(f"Test top-1 accuracy: {global_accuracy:.4f}")
    print()

    # Position baseline
    position_to_card, position_counters = train_position_baseline(train_sequences)
    position_accuracy = evaluate_position_baseline(test_sequences, position_to_card)

    print("=== Position-Based Frequency Baseline ===")
    for pos in [1, 2, 3, 14, 15, 16, 29, 30, 31, 44]:
        print(
            f"Position {pos}: predict {position_to_card[pos]} "
            f"(train count={position_counters[pos][position_to_card[pos]]})"
        )
    print(f"Test top-1 accuracy: {position_accuracy:.4f}")
    print()

    # Bigram baseline
    next_card_map, bigram_counters = train_bigram_baseline(train_sequences)
    bigram_accuracy = evaluate_bigram_baseline(
        test_sequences,
        next_card_map,
        fallback_card=most_common_card,
    )

    print("=== Bigram Baseline ===")
    sample_prev_cards = [
        "Lich's_Caress",
        "Luminous_Bonds",
        "Shock",
        "Skyscanner",
        "Swamp_4",
    ]
    for prev_card in sample_prev_cards:
        if prev_card in next_card_map:
            predicted = next_card_map[prev_card]
            count = bigram_counters[prev_card][predicted]
            print(f"After {prev_card}: predict {predicted} (train count={count})")
        else:
            print(f"After {prev_card}: unseen in training, would fall back to {most_common_card}")

    print(f"Test top-1 accuracy: {bigram_accuracy:.4f}")