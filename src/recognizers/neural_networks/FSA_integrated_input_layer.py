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
    It computes word embeddings, adds positional encodings,
    computes FSA state embeddings for each position in the input sequence,
    concatenates the word/position embeddings with the FSA state embeddings,
    and finally projects the result to the model's hidden dimension.
    """

    def __init__(
        self,
        word_vocab: Any,
        word_embedding_dim: int,
        fsa_embedding_dim: int,
        output_dim: int,
        fsa: FSA,
        use_padding: bool,
        dropout: float | None = None,
    ):
        """
        Args:
            word_vocab: The vocabulary object for the main model. Expected to have
                        `__len__` and `to_string(int)` methods.
            word_embedding_dim: The dimension of word embeddings.
            fsa_embedding_dim: The dimension of FSA state embeddings.
            output_dim: The final output dimension of the layer.
            fsa: A rayuela.fsa.fsa.FSA object.
            use_padding: Whether to use padding for word embeddings.
            dropout: Dropout rate.
        """
        super().__init__()

        self.word_vocab = word_vocab
        vocabulary_size = len(self.word_vocab)

        # 1. Extract FSA information and build mappings
        states = fsa.ordered_states
        state_to_id = {state: i for i, state in enumerate(states)}
        num_states = len(states)

        start_state_obj, _ = next(fsa.I)
        self.start_state_id = state_to_id[start_state_obj]

        fsa_alphabet = {str(sym): sym for sym in fsa.Sigma}

        # 2. Build the expanded transition matrix for the main vocabulary
        expanded_transitions = torch.arange(
            num_states, dtype=torch.long
        ).unsqueeze(1).expand(num_states, vocabulary_size)

        for word_idx in range(vocabulary_size):
            symbol_str = self.word_vocab.to_string(word_idx)
            if symbol_str in fsa_alphabet:
                fsa_sym = fsa_alphabet[symbol_str]
                for p_state in states:
                    p_id = state_to_id[p_state]
                    if fsa_sym in fsa.δ[p_state]:
                        q_state = next(iter(fsa.δ[p_state][fsa_sym]))
                        q_id = state_to_id[q_state]
                        expanded_transitions[p_id, word_idx] = q_id
        
        self.register_buffer("fsa_transitions", expanded_transitions)

        # 3. Initialize layers
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
        device = word_id_sequence.device
        batch_size, sequence_length = word_id_sequence.shape

        # 1. Word embeddings with positional encoding
        word_embeds = self.word_embedding(word_id_sequence)
        word_embeds_with_pos = self.positional_encoding.forward_from_position(
            word_embeds, 0
        )

        # 2. Compute FSA state sequence
        current_states = torch.full(
            (batch_size,), self.start_state_id, dtype=torch.long, device=device
        )
        fsa_state_ids = torch.empty(
            batch_size, sequence_length, dtype=torch.long, device=device
        )

        local_fsa_transitions = self.fsa_transitions.to(device)

        for i in range(sequence_length):
            input_symbols = word_id_sequence[:, i]
            current_states = local_fsa_transitions[current_states, input_symbols]
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
