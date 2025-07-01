import torch
import math
from rau.unidirectional import Unidirectional
from rau.tools.torch.embedding_layer import EmbeddingLayer

class ResettablePositionalInputLayer(Unidirectional):
    """
    提案手法「記号でリセットされる局所的位置エンコーディング」を実装した
    Transformerの入力層です。トークンの埋め込みと、リセット可能な正弦波
    位置エンコーディングを組み合わせます。
    """

    def __init__(
        self,
        vocabulary_size: int,
        d_model: int,
        reset_symbol_ids: set[int],
        dropout: float = 0.1,
        use_padding: bool = False,
        shared_embeddings: torch.Tensor | None = None,
        max_len: int = 5000
    ):
        super().__init__()
        self.embedding = EmbeddingLayer(
            vocabulary_size,
            d_model,
            use_padding=use_padding,
            shared_embeddings=shared_embeddings
        )
        self.d_model = d_model
        self.reset_symbol_ids = reset_symbol_ids
        self.dropout = torch.nn.Dropout(p=dropout)

        # 計算効率化のために、位置エンコーディングの値を事前計算しておく
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, input_sequence: torch.Tensor, **kwargs):
        # input_sequence の形状: (バッチサイズ, 系列長)
        batch_size, seq_len = input_sequence.size()

        # ステップ1 & 2: 局所的な位置番号の割り当てとリセット
        local_positions = torch.zeros_like(input_sequence, dtype=torch.long)
        current_pos = torch.zeros(batch_size, device=input_sequence.device, dtype=torch.long)

        for i in range(seq_len):
            token_ids = input_sequence[:, i]
            local_positions[:, i] = current_pos

            # リセット記号があるかチェック
            is_reset = torch.zeros_like(token_ids, dtype=torch.bool)
            for reset_sym_id in self.reset_symbol_ids:
                is_reset |= (token_ids == reset_sym_id)

            # 位置カウンターを更新（リセット記号の箇所は0に、それ以外はインクリメント）
            current_pos = (current_pos + 1) * (~is_reset)

        # ステップ3: 位置情報のベクトル化
        # 事前計算したテーブルから位置ベクトルを取得
        pos_encodings = self.pe[local_positions]

        # トークンの埋め込みベクトルを取得
        embeddings = self.embedding(input_sequence) * math.sqrt(self.d_model)

        # 位置ベクトルを埋め込みベクトルに加算し、ドロップアウトを適用
        output = self.dropout(embeddings + pos_encodings)

        return output