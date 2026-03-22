# data.py
#takes the messy csv data and turns it into id sequences for da transformer
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

import pandas as pd
import torch
from torch.utils.data import Dataset


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


@dataclass
class Vocab:
    token_to_id: Dict[str, int]
    id_to_token: Dict[int, str]

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    @property
    def size(self) -> int:
        return len(self.token_to_id)


def load_draft_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    if df.shape[1] != 10:
        raise ValueError(f"Expected 10 columns in {path}, got {df.shape[1]}")
    df.columns = ["draft_id", "set_code"] + [f"seat_{i}" for i in range(1, 9)]
    return df


def load_card_metadata_safe(
    rating_path: str,
    land_rating_path: Optional[str] = None,
) -> pd.DataFrame:
    def _read_one(path: str) -> pd.DataFrame:
        df = pd.read_csv(path, sep="\t")
        df.columns = [col.strip() for col in df.columns]
        if "Rating" in df.columns:
            df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
        return df

    ratings_df = _read_one(rating_path)

    if land_rating_path is not None:
        land_df = _read_one(land_rating_path)
        combined = pd.concat([ratings_df, land_df], ignore_index=True)
    else:
        combined = ratings_df

    return combined


def build_valid_card_name_set(metadata_df: pd.DataFrame) -> Set[str]:
    return set(metadata_df["Name"].astype(str).tolist())


def parse_seat_sequence(seat_string: str, valid_names: Set[str]) -> List[str]:
    """
    Parse one seat string into card names, correctly handling card names
    that contain commas, such as 'Sai,_Master_Thopterist'.

    Strategy:
    - Split naively on commas
    - Greedily merge adjacent pieces until they match a valid card name
    """
    if not isinstance(seat_string, str):
        return []

    parts = [p.strip() for p in seat_string.split(",") if p.strip()]
    parsed = []

    i = 0
    while i < len(parts):
        current = parts[i]

        if current in valid_names:
            parsed.append(current)
            i += 1
            continue

        # Greedily merge with following pieces until valid
        j = i + 1
        found = None
        candidate = current

        while j < len(parts):
            candidate = candidate + "," + parts[j]
            if candidate in valid_names:
                found = candidate
                break
            j += 1

        if found is not None:
            parsed.append(found)
            i = j + 1
        else:
            # Keep original piece so problems are visible during debugging
            parsed.append(current)
            i += 1

    return parsed


def extract_seat_sequences(df: pd.DataFrame, valid_names: Set[str]) -> List[List[str]]:
    seat_cols = [f"seat_{i}" for i in range(1, 9)]
    all_seats = df[seat_cols].values.flatten()

    sequences = []
    for seat_string in all_seats:
        seq = parse_seat_sequence(seat_string, valid_names)
        if seq:
            sequences.append(seq)

    return sequences


def validate_sequence_lengths(
    sequences: List[List[str]],
    expected_len: int = 45,
    verbose: bool = True,
) -> None:
    lengths = pd.Series([len(seq) for seq in sequences])
    if verbose:
        print("Sequence length counts:")
        print(lengths.value_counts().sort_index())

    invalid = lengths[lengths != expected_len]
    if len(invalid) > 0:
        print(f"Warning: found {len(invalid)} sequences not equal to {expected_len}.")
    else:
        print(f"All sequences match expected length {expected_len}.")


def build_vocab(sequences: List[List[str]], min_freq: int = 1) -> Vocab:
    token_counts: Dict[str, int] = {}

    for seq in sequences:
        for token in seq:
            token_counts[token] = token_counts.get(token, 0) + 1

    token_to_id = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }

    for token, count in sorted(token_counts.items()):
        if count >= min_freq:
            token_to_id[token] = len(token_to_id)

    id_to_token = {idx: token for token, idx in token_to_id.items()}
    return Vocab(token_to_id=token_to_id, id_to_token=id_to_token)


def encode_sequence(sequence: List[str], vocab: Vocab) -> List[int]:
    return [vocab.token_to_id.get(token, vocab.unk_id) for token in sequence]


def encode_sequences(sequences: List[List[str]], vocab: Vocab) -> List[List[int]]:
    return [encode_sequence(seq, vocab) for seq in sequences]


class DraftDataset(Dataset):
    def __init__(self, encoded_sequences: List[List[int]], seq_len: int = 45):
        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for seq in encoded_sequences:
            if len(seq) != seq_len:
                continue

            x = torch.tensor(seq[:-1], dtype=torch.long)
            y = torch.tensor(seq[1:], dtype=torch.long)
            self.samples.append((x, y))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def build_metadata_lookup(metadata_df: pd.DataFrame) -> Dict[str, dict]:
    lookup: Dict[str, dict] = {}

    for _, row in metadata_df.iterrows():
        name = row["Name"]
        lookup[name] = {
            "casting_cost_1": row.get("Casting Cost 1"),
            "casting_cost_2": row.get("Casting Cost 2"),
            "card_type": row.get("Card Type"),
            "rarity": row.get("Rarity"),
            "rating": row.get("Rating"),
        }

    return lookup


def load_train_and_test_sequences(
    train_csv_path: str,
    test_csv_path: str,
    rating_path: str,
    land_rating_path: Optional[str] = None,
    expected_len: int = 45,
    verbose: bool = True,
) -> Tuple[List[List[str]], List[List[str]], pd.DataFrame]:
    metadata_df = load_card_metadata_safe(rating_path, land_rating_path)
    valid_names = build_valid_card_name_set(metadata_df)

    train_df = load_draft_csv(train_csv_path)
    test_df = load_draft_csv(test_csv_path)

    train_sequences = extract_seat_sequences(train_df, valid_names)
    test_sequences = extract_seat_sequences(test_df, valid_names)

    if verbose:
        print(f"Loaded {len(train_sequences)} train seat sequences.")
        print(f"Loaded {len(test_sequences)} test seat sequences.")
        validate_sequence_lengths(train_sequences, expected_len=expected_len, verbose=True)
        validate_sequence_lengths(test_sequences, expected_len=expected_len, verbose=True)

    return train_sequences, test_sequences, metadata_df


def decode_ids(ids: List[int], vocab: Vocab) -> List[str]:
    return [vocab.id_to_token.get(idx, UNK_TOKEN) for idx in ids]


if __name__ == "__main__":
    TRAIN_CSV = "train.csv"
    TEST_CSV = "test.csv"
    RATING_TSV = "m19_rating.tsv"
    LAND_RATING_TSV = "m19_land_rating.tsv"

    train_sequences, test_sequences, metadata_df = load_train_and_test_sequences(
        TRAIN_CSV,
        TEST_CSV,
        RATING_TSV,
        LAND_RATING_TSV,
        expected_len=45,
        verbose=True,
    )

    vocab = build_vocab(train_sequences)
    print(f"Vocab size: {vocab.size}")

    train_encoded = encode_sequences(train_sequences, vocab)
    test_encoded = encode_sequences(test_sequences, vocab)

    train_dataset = DraftDataset(train_encoded, seq_len=45)
    test_dataset = DraftDataset(test_encoded, seq_len=45)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    x, y = train_dataset[0]
    print("Example input IDs:", x[:10].tolist())
    print("Example target IDs:", y[:10].tolist())
    print("Decoded input:", decode_ids(x[:10].tolist(), vocab))
    print("Decoded target:", decode_ids(y[:10].tolist(), vocab))