import math
from typing import Any

import torch
from torch import nn

from rayuela.fsa.fsa import FSA
from rau.models.transformer.input_layer import (
    ScaledEmbeddingLayer,
    SinusoidalPositionalEncodingLayer,
)
from rau.unidirectional import Unidirectional
from recognizers.automata.finite_automaton import (
    FiniteAutomatonContainer
)

class FSAStateEmbedding(nn.Module):
    """
    An embedding layer for FSA states.
    Treats state IDs as a vocabulary and learns a vector for each state.
    """

    def __init__(self, num_states: int, embedding_dim: int):
        """
        Args:
            num_states: The total number of states in the FSA.
            embedding_dim: The dimension of the state embeddings.
        """
        super().__init__()
        self.embedding = nn.Embedding(num_states, embedding_dim)

    def forward(self, state_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_ids: A tensor of state IDs.
        Returns:
            A tensor of state embeddings.
        """
        return self.embedding(state_ids)


class FSAIntegratedInputLayer(Unidirectional):
    """
    An input layer that integrates FSA state information.
    """

    def __init__(
        self,
        word_vocab: Any,
        fsa_alphabet: list[str],
        fsa_container: FiniteAutomatonContainer,
        word_embedding_dim: int,
        fsa_embedding_dim: int,
        output_dim: int,
        use_padding: bool,
        dropout: float | None = None,
    ):
        super().__init__()

        self.word_vocab = word_vocab
        vocabulary_size = len(self.word_vocab)
        num_states = fsa_container.num_states()
        self.start_state_id = fsa_container.initial_state()

        # Build transition tensor from the container
        fsa_symbol_to_id = {symbol: i for i, symbol in enumerate(fsa_alphabet)}
        fsa_transitions = torch.arange(
            num_states, dtype=torch.long
        ).unsqueeze(1).expand(num_states, len(fsa_alphabet))

        for t in fsa_container.transitions():
            fsa_transitions[t.state_from, t.symbol] = t.state_to

        expanded_transitions = torch.arange(
            num_states, dtype=torch.long
        ).unsqueeze(1).expand(num_states, vocabulary_size)

        for word_idx in range(vocabulary_size):
            symbol_str = self.word_vocab.to_string(word_idx)
            if symbol_str in fsa_symbol_to_id:
                fsa_symbol_id = fsa_symbol_to_id[symbol_str]
                expanded_transitions[:, word_idx] = fsa_transitions[:, fsa_symbol_id]

        self.register_buffer("fsa_transitions", expanded_transitions)

        # Initialize layers
        self.word_embedding = ScaledEmbeddingLayer(
            vocabulary_size=vocabulary_size,
            output_size=word_embedding_dim,
            use_padding=use_padding,
            shared_embeddings=None,
        )
        self.positional_encoding = SinusoidalPositionalEncodingLayer()
        self.fsa_state_embedding = FSAStateEmbedding(
            num_states=num_states,
            embedding_dim=fsa_embedding_dim,
        )
        self.projection = nn.Linear(word_embedding_dim + fsa_embedding_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout) if dropout is not None else None

    def forward(self, word_id_sequence: torch.Tensor, **kwargs) -> torch.Tensor:

        batch_size, sequence_length = word_id_sequence.shape
        device = word_id_sequence.device

        # 1. Word embeddings with positional encoding
        # Modules are already on the correct device from the main .to(device) call
        word_embeds = self.word_embedding(word_id_sequence)
        word_embeds_with_pos = self.positional_encoding.forward_from_position(
            word_embeds, 0
        )

        # 2. Compute FSA state sequence
        # self.fsa_transitions is a buffer and is already on the correct device
        current_states = torch.full(
            (batch_size,), self.start_state_id, dtype=torch.long, device=device
        )
        fsa_state_ids = torch.empty(
            batch_size, sequence_length, dtype=torch.long, device=device
        )

        for i in range(sequence_length):
            input_symbols = word_id_sequence[:, i]
            current_states = self.fsa_transitions[current_states, input_symbols]
            fsa_state_ids[:, i] = current_states

        # 3. Get FSA state embeddings
        fsa_state_embeds = self.fsa_state_embedding(fsa_state_ids)

        # 4. Concatenate
        combined_embeds = torch.cat([word_embeds_with_pos, fsa_state_embeds], dim=-1)

        # 5. Project to output dimension and apply dropout
        projected_embeds = self.projection(combined_embeds)

        if self.dropout is not None:
            projected_embeds = self.dropout(projected_embeds)

        return projected_embeds
