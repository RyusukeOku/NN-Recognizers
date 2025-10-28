import argparse
import pathlib
import sys
import torch
from typing import List, Tuple, Iterable

# EDSMアルゴリズムをインポート
from src.recognizers.automata.GsmAlgorithms import run_EDSM
# AALpyの基本オートマトン型 (run_EDSMの戻り値)
from src.recognizers.automata.automaton_aalpy import DeterministicAutomaton
# NNが使用するRayuelaベースのコンテナ
from src.recognizers.automata.finite_automaton import FiniteAutomatonContainer, FiniteAutomatonTransition
from rayuela.base.state import State
# NNのボキャブラリをロードするため
from rau.vocab import ToStringVocabulary
from src.recognizers.neural_networks.vocabulary import load_vocabulary_data_from_file

def load_learning_data(
    main_tok_path: pathlib.Path, label_txt_path: pathlib.Path
) -> List[Tuple[Tuple[str, ...], bool]]:
    """
    main.tok と labels.txt から (トークン列, ラベル) のリストを作成します。
    run_EDSM が 'labeled_sequences' として要求する形式です。
    """
    learning_data = []
    if not main_tok_path.is_file():
        print(f"エラー: {main_tok_path} が見つかりません。", file=sys.stderr)
        sys.exit(1)
    if not label_txt_path.is_file():
        print(f"エラー: {label_txt_path} が見つかりません。", file=sys.stderr)
        sys.exit(1)

    with main_tok_path.open() as f_tok, label_txt_path.open() as f_labels:
        for line_tok, line_label in zip(f_tok, f_labels):
            tokens = tuple(line_tok.strip().split())
            try:
                label = bool(int(line_label.strip()))
            except ValueError:
                print(f"エラー: labels.txt に不正な行があります: {line_label.strip()}", file=sys.stderr)
                sys.exit(1)
            learning_data.append((tokens, label))
    return learning_data

def load_vocabulary(vocab_path: pathlib.Path) -> ToStringVocabulary:
    """
    main.vocab ファイルから ToStringVocabulary をロードします。
    """
    if not vocab_path.is_file():
        print(f"エラー: {vocab_path} が見つかりません。", file=sys.stderr)
        print("先に prepare_data.py を実行してください。", file=sys.stderr)
        sys.exit(1)
    
    vocab_data = load_vocabulary_data_from_file(vocab_path)
    # EDSMはBOS/EOSを扱わないため、use_bos=False, use_eos=False でロード
    return ToStringVocabulary(
        vocab_data.tokens,
        include_bos=False,
        include_eos=False,
        include_unk=vocab_data.allow_unk
    )

def convert_aalpy_dfa_to_container(
    dfa: DeterministicAutomaton, vocab: ToStringVocabulary
) -> FiniteAutomatonContainer:
    """
    AALpy の DeterministicAutomaton (Dfa) オブジェクトを
    FSAIntegratedInputLayer が要求する FiniteAutomatonContainer に変換します。
    (rpni_learner.py の _convert_aalpy_dfa_to_container と同等の機能)
    """
    
    # 状態IDを 0 から N-1 にマッピング
    aalpy_states = sorted(dfa.states, key=lambda s: s.state_id)
    state_map = {state: i for i, state in enumerate(aalpy_states)}
    
    if dfa.initial_state not in state_map:
        raise ValueError("学習されたDFAの初期状態が状態リストに含まれていません。")

    initial_state_id = state_map[dfa.initial_state]
    
    # vocab に含まれるシンボルのみをアルファベットサイズとする
    # (BOS/EOS/UNKは除く、またはEDSM学習時のアルファベットに基づくべき)
    # ここでは vocab に含まれる全シンボル数を使う
    alphabet_size = len(vocab) 

    container = FiniteAutomatonContainer(
        num_states=len(aalpy_states),
        alphabet_size=alphabet_size, # この値はレイヤでは直接使われない
        initial_state=initial_state_id
    )

    for aalpy_state, state_id in state_map.items():
        # EDSM(DFA)では、状態が受理かどうかでラベルが決まる
        if aalpy_state.is_accepting:
            container.add_accept_state(State(state_id))

        for symbol_str, target_aalpy_state in aalpy_state.transitions.items():
            try:
                # 文字列シンボルをNNのボキャブラリIDに変換
                symbol_id = vocab.to_int(symbol_str)
            except KeyError:
                # 学習データにはあったが、main.vocab にないシンボル (BOS/EOSなど)
                # または prepare_data.py でUNKにまとめられたシンボル
                if vocab.unk_index is not None:
                    symbol_id = vocab.unk_index
                else:
                    print(f"警告: EDSMが学習したシンボル '{symbol_str}' が "
                          f"ボキャブラリにありません。この遷移をスキップします。", file=sys.stderr)
                    continue
            
            if target_aalpy_state not in state_map:
                print(f"警告: 遷移先の状態 {target_aalpy_state.state_id} が "
                      f"状態マップにありません。スキップします。", file=sys.stderr)
                continue

            target_state_id = state_map[target_aalpy_state]

            transition = FiniteAutomatonTransition(
                state_from=state_id,
                symbol=symbol_id,
                state_to=target_state_id,
                weight=0.0 # 重みなし
            )
            container.add_transition(transition)

    return container

def get_alphabet_from_vocab(vocab: ToStringVocabulary) -> List[str]:
    """
    FSAIntegratedInputLayer が必要とするアルファベット(文字列リスト)を
    ボキャブラリから取得します。
    """
    # BOS/EOS/UNK を除く、EDSM学習に使われたシンボルリスト
    return [t for t in vocab.all_tokens 
            if t not in (vocab.bos_string, vocab.eos_string, vocab.unk_string)]

def main():
    parser = argparse.ArgumentParser(
        description='Learn a DFA with EDSM and save it as a FiniteAutomatonContainer.'
    )
    parser.add_argument(
        '--training-data', type=pathlib.Path, required=True,
        help='Directory containing main.tok, labels.txt, and main.vocab'
    )
    parser.add_argument(
        '--output-file', type=pathlib.Path,
        help='Path to save the learned automaton (default: <training-data>/edsm_automaton.pt)'
    )
    parser.add_argument(
        '--delta', type=float, default=0.005,
        help='Confidence parameter (delta) for Hoeffding bound (default: 0.005)'
    )
    args = parser.parse_args()

    main_tok_path = args.training_data / 'main.tok'
    labels_path = args.training_data / 'labels.txt'
    vocab_path = args.training_data / 'main.vocab'
    
    output_path = args.output_file
    if not output_path:
        output_path = args.training_data / 'edsm_automaton.pt'
    
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. ボキャブラリをロード (シンボル -> ID 変換のため)
    print(f"Loading vocabulary from {vocab_path}...")
    vocab = load_vocabulary(vocab_path)
    
    # 2. 学習データをロード (シンボルは文字列のまま)
    print(f"Loading data from {main_tok_path} and {labels_path}...")
    learning_data = load_learning_data(main_tok_path, labels_path)
    
    # 3. EDSM (DFA) を学習
    print(f"Running EDSM with delta={args.delta}...")
    # GsmAlgorithms.py の run_EDSM は AALpy の DeterministicAutomaton を返す
    learned_aalpy_dfa = run_EDSM(
        data=learning_data,
        automaton_type='dfa', # DFAを学習
        delta=args.delta,
        print_info=True
    )

    if learned_aalpy_dfa is None:
        print("エラー: EDSMによる学習に失敗しました。", file=sys.stderr)
        sys.exit(1)

    print(f"Learned AALpy DFA with {len(learned_aalpy_dfa.states)} states.")

    # 4. AALpy DFA -> FiniteAutomatonContainer に変換
    print("Converting AALpy DFA to FiniteAutomatonContainer...")
    fsa_container = convert_aalpy_dfa_to_container(learned_aalpy_dfa, vocab)
    print(f"Conversion complete. Container has {fsa_container.num_states()} states.")

    # 5. NNレイヤが必要とするアルファベットリストを取得
    alphabet_list = get_alphabet_from_vocab(vocab)

    # 6. 保存
    save_data = {
        'automaton': fsa_container,
        'alphabet': alphabet_list
    }
    torch.save(save_data, output_path)
    print(f"Learned automaton and alphabet saved to {output_path}")

if __name__ == '__main__':
    main()