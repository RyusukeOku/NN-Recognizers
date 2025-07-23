import torch
import torch.nn as nn
import torch.nn.functional as F
from rau.models.rnn import LSTM

class NeuralLBA(nn.Module):
    """
    ニューラル線形拘束オートマトン (Neural Linear Bounded Automaton)
    """
    def __init__(self, input_size, hidden_size, n_steps, device):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_steps = n_steps
        self.device = device

        self.controller = LSTM(input_size, hidden_size)
        
        # ヘッドの移動（左、維持、右）、書き込みベクトル、書き込みゲートを生成するための線形層
        self.controller_to_tape_op = nn.Linear(hidden_size, input_size + 3 + 1)
        
    def forward(self, embedded_input):
        """
        Args:
            embedded_input (torch.Tensor): 埋め込み後の入力系列 (batch_size, seq_len, input_size)
        """
        batch_size, seq_len, _ = embedded_input.size()
        
        # テープの初期化
        tape = embedded_input.clone()
        
        # ヘッド位置の初期化 (バッチ内の各シーケンスの先頭を指す)
        head_pos = torch.zeros(batch_size, seq_len, device=self.device)
        head_pos[:, 0] = 1.0
        
        # コントローラの初期状態
        controller_state = self.controller.initial_state(batch_size, self.device)

        for _ in range(self.n_steps):
            # 1. テープから情報を読み取る (加重平均)
            # (batch_size, seq_len, input_size) * (batch_size, seq_len, 1) -> (batch_size, input_size)
            read_vec = (tape * head_pos.unsqueeze(-1)).sum(dim=1)
            
            # 2. 読み取ったベクトルをコントローラLSTMに入力
            controller_output, controller_state = self.controller.next_with_output(read_vec, controller_state)

            # 3. コントローラの出力からテープ操作を生成
            tape_ops = self.controller_to_tape_op(controller_output)
            write_vec = tape_ops[:, :self.input_size]
            move_logits = tape_ops[:, self.input_size:self.input_size + 3]
            write_gate = torch.sigmoid(tape_ops[:, self.input_size + 3:])
            
            # 4. Gumbel-Softmaxでヘッドを移動
            # (batch_size, 3) -> (batch_size, seq_len)
            move_dist = F.gumbel_softmax(move_logits, tau=1, hard=True)
            
            # ヘッド位置の更新
            # 畳み込みを利用してヘッドをシフト
            move_filters = torch.tensor([[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0]]], device=self.device).repeat(batch_size,1,1)
            
            current_head_pos_padded = F.pad(head_pos.unsqueeze(1), (1, 1))

            new_head_pos = F.conv1d(current_head_pos_padded, move_filters, groups=batch_size).squeeze(1)

            head_pos = new_head_pos

            # 5. テープを書き換える
            # (batch_size, 1, input_size) * (batch_size, seq_len, 1) -> (batch_size, seq_len, input_size)
            write_update = write_vec.unsqueeze(1) * head_pos.unsqueeze(-1)
            tape = tape * (1 - write_gate.unsqueeze(1)) + write_update * write_gate.unsqueeze(1)

        # 最終ステップのコントローラの隠れ状態を出力
        return controller_state.hidden