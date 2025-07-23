import torch
import torch.nn as nn
import torch.nn.functional as F

class NgramHead(nn.Module):
    def __init__(self, n: int, d_model: int):
        super().__init__()
        if n <= 0:
            raise ValueError("n-gram size 'n' must be a positive integer.")
        self.n = n
        self.d_model = d_model
        self.W1 = nn.Linear(d_model, d_model)
        self.W2 = nn.Linear(d_model, d_model)

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.size()

        if seq_len < self.n:
            return self.W1(hidden_states)

        input_seq_len = input_ids.size(1)
        
        if input_seq_len < self.n:
            attention_mask = torch.zeros(
                (batch_size, seq_len, seq_len), 
                device=hidden_states.device, 
                dtype=torch.bool
            )
        else:
            ngrams = input_ids.unfold(dimension=1, size=self.n, step=1)
            padded_ngrams = F.pad(ngrams, (0, 0, self.n - 1, 0))
            
            ngrams_a = padded_ngrams.unsqueeze(2)
            ngrams_b = padded_ngrams.unsqueeze(1)
    
            attention_mask = (ngrams_a == ngrams_b).all(dim=-1)

        current_mask_seq_len = attention_mask.size(1)
        if current_mask_seq_len != seq_len:
            pad_diff = seq_len - current_mask_seq_len
            attention_mask = F.pad(attention_mask, (0, pad_diff, 0, pad_diff), "constant", 0)

        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=hidden_states.device, dtype=torch.bool), diagonal=-1)
        
        attention_mask = attention_mask & causal_mask.unsqueeze(0)

        num_attended = attention_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        
        pooled_hidden_states = torch.matmul(attention_mask.float() / num_attended, hidden_states)

        direct_path = self.W1(hidden_states)
        ngram_path = self.W2(pooled_hidden_states)

        return direct_path + ngram_path