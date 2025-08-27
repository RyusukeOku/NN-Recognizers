import math
from typing import Any

import torch
from torch import nn

from recognizers.automata.finite_automaton import FiniteAutomatonContainer
from rau.vocab import Vocab
from rau.models.transformer.input_layer import (
    ScaledEmbeddingLayer,
    SinusoidalPositionalEncodingLayer,
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


class FSAIntegratedInputLayer(nn.Module):
    """
    An input layer that integrates FSA state information.
    It computes word embeddings, adds positional encodings,
    computes FSA state embeddings for each position in the input sequence,
    and concatenates the word/position embeddings with the FSA state embeddings.
    """

    def __init__(
        self,
        word_vocab: Vocab,
        word_embedding_dim: int,
        fsa_embedding_dim: int,
        fsa_container: FiniteAutomatonContainer,
        use_padding: bool,
        dropout: float | None = None,
    ):
        """
        Args:
            word_vocab: The vocabulary object for the main model.
            word_embedding_dim: The dimension of word embeddings.
            fsa_embedding_dim: The dimension of FSA state embeddings.
            fsa_container: A container for the FSA, expected to have:
                - `num_states` (int): The total number of states.
                - `start_state` (int): The ID of the start state.
                - `transitions` (torch.Tensor): A tensor of shape
                  (num_states, fsa_vocab_size) representing the transition function.
                - `alphabet` (Dict[str, int]): A mapping from symbols to FSA vocab IDs.
            use_padding: Whether to use padding for word embeddings.
            dropout: Dropout rate.
        """
        super().__init__()

        # Basic validation of the fsa_container object
        for attr in ["num_states", "start_state", "transitions", "alphabet"]:
            if not hasattr(fsa_container, attr):
                raise ValueError(f"The FSA container object must have a '{attr}' attribute.")
        if not isinstance(fsa_container.transitions, torch.Tensor):
            raise ValueError("The FSA container's 'transitions' attribute must be a torch.Tensor.")
        if not isinstance(fsa_container.alphabet, dict):
            raise ValueError("The FSA container's 'alphabet' attribute must be a dict.")

        self.fsa = fsa_container
        self.word_vocab = word_vocab
        vocabulary_size = len(self.word_vocab)

        self.word_embedding = ScaledEmbeddingLayer(
            vocabulary_size=vocabulary_size,
            output_size=word_embedding_dim,
            use_padding=use_padding,
            shared_embeddings=None,
        )
        self.positional_encoding = SinusoidalPositionalEncodingLayer()

        self.fsa_state_embedding = FSAStateEmbedding(
            num_states=self.fsa.num_states,
            embedding_dim=fsa_embedding_dim,
        )

        self.dropout = nn.Dropout(p=dropout) if dropout is not None else None

        # --- Build the expanded transition matrix ---
        # This matrix maps states and main vocabulary IDs to next states.

        # Default transition is a self-loop.
        expanded_transitions = torch.arange(
            self.fsa.num_states, dtype=torch.long
        ).unsqueeze(1).expand(self.fsa.num_states, vocabulary_size)

        fsa_alphabet = self.fsa.alphabet
        fsa_internal_transitions = self.fsa.transitions.long()

        for word_idx in range(vocabulary_size):
            # Assuming a vocab interface to get string from index.
            # Based on `rau.vocab`, this seems to be `i2s`.
            symbol_str = self.word_vocab.i2s[word_idx]
            if symbol_str in fsa_alphabet:
                fsa_symbol_idx = fsa_alphabet[symbol_str]
                # Copy the transitions for this symbol from the original FSA matrix.
                expanded_transitions[:, word_idx] = fsa_internal_transitions[:, fsa_symbol_idx]

        self.register_buffer("fsa_transitions", expanded_transitions)


    def forward(self, word_id_sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            word_id_sequence: A tensor of word IDs of shape (batch_size, sequence_length).
        Returns:
            A tensor of combined embeddings of shape
            (batch_size, sequence_length, word_embedding_dim + fsa_embedding_dim).
        """
        batch_size, sequence_length = word_id_sequence.shape
        device = word_id_sequence.device

        # 1. Word embeddings with positional encoding
        word_embeds = self.word_embedding(word_id_sequence)
        word_embeds_with_pos = self.positional_encoding.forward_from_position(
            word_embeds, 0
        )

        # 2. Compute FSA state sequence
        current_states = torch.full(
            (batch_size,), self.fsa.start_state, dtype=torch.long, device=device
        )
        fsa_state_ids = torch.empty(
            batch_size, sequence_length, dtype=torch.long, device=device
        )

        for i in range(sequence_length):
            input_symbols = word_id_sequence[:, i]
            # Get next states for the whole batch at once
            current_states = self.fsa_transitions[current_states, input_symbols]
            fsa_state_ids[:, i] = current_states

        # 3. Get FSA state embeddings
        fsa_state_embeds = self.fsa_state_embedding(fsa_state_ids)

        # 4. Concatenate and apply dropout
        combined_embeds = torch.cat([word_embeds_with_pos, fsa_state_embeds], dim=-1)

        if self.dropout is not None:
            combined_embeds = self.dropout(combined_embeds)

        return combined_embeds
