# model.py

from __future__ import annotations

import torch
import torch.nn as nn


class DraftTransformer(nn.Module):
    """
    Autoregressive Transformer for Magic draft sequence modeling.

    Input:
        x: LongTensor of shape (batch_size, seq_len)

    Output:
        logits: FloatTensor of shape (batch_size, seq_len, vocab_size)

    For each position t, logits[:, t, :] predicts the next card target at that position.
    """

    def __init__(
        self,
        vocab_size: int,
        seq_len: int = 44,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.pad_id = pad_id

        # Token embedding: maps card IDs -> dense vectors
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=pad_id,
        )

        # Positional embedding: tells the model where in the draft sequence it is
        self.position_embedding = nn.Embedding(
            num_embeddings=seq_len,
            embedding_dim=d_model,
        )

        # Transformer encoder block
        # We use an encoder with a causal mask to make it autoregressive.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        # Final layer norm before output projection
        self.final_ln = nn.LayerNorm(d_model)

        # Output head: maps hidden states -> logits over vocabulary
        self.output_head = nn.Linear(d_model, vocab_size)

    def make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create an upper-triangular causal mask so position t cannot attend to future positions.

        Shape:
            (seq_len, seq_len)

        PyTorch Transformer expects:
            True = masked / blocked
            False = allowed
        """
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: LongTensor of shape (batch_size, seq_len)

        Returns:
            logits: FloatTensor of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = x.shape

        if seq_len > self.seq_len:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds configured max seq_len {self.seq_len}"
            )

        device = x.device

        # Position IDs: [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, seq_len)

        # Embed tokens and positions
        tok_emb = self.token_embedding(x)              # (batch_size, seq_len, d_model)
        pos_emb = self.position_embedding(positions)   # (1, seq_len, d_model)

        # Combine embeddings
        h = tok_emb + pos_emb

        # Causal attention mask
        causal_mask = self.make_causal_mask(seq_len, device)

        # Optional padding mask: True where padding tokens exist
        # This matters more if you later batch variable-length sequences.
        padding_mask = (x == self.pad_id)

        # Run transformer
        h = self.transformer(
            h,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )  # (batch_size, seq_len, d_model)

        h = self.final_ln(h)

        # Project to vocabulary logits
        logits = self.output_head(h)  # (batch_size, seq_len, vocab_size)

        return logits


if __name__ == "__main__":
    # Simple smoke test
    batch_size = 4
    seq_len = 44
    vocab_size = 283

    model = DraftTransformer(
        vocab_size=vocab_size,
        seq_len=seq_len,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=512,
        dropout=0.1,
        pad_id=0,
    )

    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", logits.shape)