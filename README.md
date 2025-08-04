研究に使っているレポジトリです。

# experiments

## *.bash

- submit_job.bash: 計算クラスタに，実験ジョブを投入するためのメインスクリプト
- job.bash: ここのジョブが実行する処理の本体
- setup.bash, include.bash: 実験環境の設定や共通の関数を定義するファイル
- singularity_*.bash: Singularityを操作するスクリプト

## training/

experiments/training/ ディレクトリには、モデルの訓練と評価に関連するジョブを投入するためのスクリ-プトが格納されています。

- submit_train_and_evaluate_jobs.bash:
    - モデルの訓練と、その後の評価をまとめて行う一連のジョブを投入します。
- submit_rerun_evaluate_jobs.bash:
    - 既に訓練済みのモデルに対して、評価のみを再実行するジョブを投入します。これは、評価データや評価方法を変更して再度テストしたい場合に利用します。

## dataset_generation/

experiments/dataset_generation/ ディレクトリには、実験に使用する様々なデータセットを生成するためのジョブを投入するスクリプトが格納されています。

- submit_prepare_automaton_language_jobs.bash:
    - 形式的なオートマトンから言語データを生成するジョブを投入します。
- submit_prepare_hand_coded_language_jobs.bash:
    - src/recognizers/hand_picked_languages/にあるような、手動でコーディングされた特定の規則を持つ言語データを生成するジョブを投入します。
- submit_prepare_test_edit_distance_jobs.bash:
    - 文字列間の編集距離を計算するテスト用のデータを生成するジョブを投入します。これは、モデルがどの程度ノイズに強いかを評価するために使われる可能性があります。

## analysis/

experiments/analysis/ ディレクトリには、訓練済みモデルの性能を分析し、結果をまとめるためのスクリプトが格納されています。

- plot_*.bash:
    - plot_cross_entropy_vs_edit_distance.bash: モデルの予測誤差（クロスエントロピー）と文字列の編集距離の関係をプロットします。
    - plot_cross_entropy_vs_length.bash: 予測誤差と入力文字列の長さの関係をプロットします。
- print_*.bash:
    - print_main_tables.bash: 実験結果の主要な部分をまとめた表を出力します（論文のメインテーブルなど）。
    - print_full_tables.bash: より詳細な結果を含む完全な表を出力します。
    - print_hardest_examples.bash: モデルが最も苦手とした（＝予測が困難だった）サンプルを出力します。

# results

results/ ディレクトリには、モデルの評価実験から得られた結果をまとめたCSVファイルが格納されています。

ファイル名から、それぞれのファイルが特定の実験条件やモデル構成に対応した評価結果の要約（特に平均値などの統計量）であることがわかります。

例えば、以下のような異なる実験の結果が保存されているようです。

- eval_results_mean_summary_lba_*.csv: LBA (Linear Bounded Automaton) に関連するモデルの評価結果。
- eval_results_mean_summary_ngram_*.csv: N-gramモデルの評価結果。
- eval_results_mean_summary_with_syntax.csv: 構文情報を利用したモデルの評価結果。
- eval_results_mean_summary_original.csv: ベースラインとなるオリジナルモデルの評価結果。

これらのファイルは、experiments/analysis/ 内のスクリプトによって読み込まれ、論文用の表やグラフを作成するために利用されると考えられます。

# scripts

scripts/ ディレクトリには、開発環境のセットアップ、コンテナの管理、依存関係のインストールなど、プロジェクト全体に関わる補助的なシェルスクリプトが格納されています。

主な役割は以下の通りです。

- コンテナ管理 (Docker & Singularity):
    - build_docker_dev_image.bash, build_singularity_image.bash: 開発や実験の再現性を保証するためのDockerやSingularityコンテナイメージをビルドします。
    - docker_shell.bash, singularity_shell.bash: ビルドしたコンテナ内でインタラクティブなシェルを起動します。
    - docker_exec.bash: コンテナ内で特定のコマンドを実行します。
    - get_docker_dev_image.bash: 事前にビルドされたDockerイメージを取得します。
    - dockerdev.bash: Docker開発環境を管理するための便利なラッパースクリプトのようです。
- 環境構築と依存関係:
    - install_python_packages.bash: pyproject.tomlなどに基づいて、必要なPythonパッケージをインストールします。
    - setup.bash: プロジェクト全体の基本的なセットアップを行います。
    - variables.bash: 他のスクリプトから参照される共通の環境変数（ファイルパス、イメージ名など）を定義しています。
- その他:
    - generate_code_submission.bash: 学会投稿などのために、ソースコードをzipファイルなどにまとめるためのスクリプトです。

# src

src/ ディレクトリは、このプロジェクトのすべてのソースコードを格納する中心的な場所です。主に3つのサブディレクトリに分かれています。

- `recognizers/`:
    - このプロジェクトの中核となるアプリケーションコードです。
    - neural_networks/: ニューラルネットワークモデルの定義、訓練、評価を行うコードが含まれています。
        1. データ準備と管理
            - prepare_data.py, prepare_data_original.py:
                - automataやhand_picked_languagesで生成された言語データを、ニューラルネットワークが読み込める形式（数値テンソルなど）に前処理・変換するスクリプトです。
                - prepare_language.bash:
                    - prepare_data.pyの実行を補助するシェルスクリプトです。
                - [data.py](http://data.py/):
                    - PyTorchのDatasetやDataLoaderのように、モデルにデータを供給するためのクラスが定義されています。
                - [vocabulary.py](http://vocabulary.py/):
                    - 文字や記号（トークン）と、それに対応する数値IDの辞書（Vocabulary）を管理します。
                - [batching.py](http://batching.py/):
                    - 長さが異なる複数の文（シーケンス）をまとめて一つのバッチにするための処理（パディングなど）を実装しています。
            1. モデルのアーキテクチャ定義
            - model_interface.py:
                - すべてのモデルが従うべき共通のインターフェース（抽象基底クラス）を定義しています。これにより、異なるモデルでも同じ訓練・評価コードを再利用できます。
            - model_interface_original.py, model_interface_ngram.py:
                - model_interfaceを継承した、具体的なモデルアーキテクチャの実装です。「オリジナルモデル」や「N-gramモデル」など、複数のモデルが試されていることがわかります。
            - [lba.py](http://lba.py/):
                - LBA (Linear Bounded Automaton) を模倣したニューラルネットワークモデルの実装の可能性があります。
            - ngram_head.py:
                - N-gramモデルの出力層（ヘッド）部分の実装です。
            - resettable_positional_encoding.py:
                - Transformerなどで使われる位置エンコーディングのカスタム実装です。「リセット可能」という名前から、特殊な処理が加えられていることが推測されます。
            1. 訓練と評価の実行
            - [train.py](http://train.py/):
                - モデルの訓練を実行するメインスクリプトです。
            - [evaluate.py](http://evaluate.py/):
                - 訓練済みモデルの性能評価を実行するメインスクリプトです。
            - training_loop.py:
                - エポックやバッチを回す訓練ループの本体ロジックが実装されています。train.pyから呼び出される形です。
            - train_and_evaluate.bash, evaluate.bash:
                - experiments/以下のジョブスクリプトから呼び出され、train.pyやevaluate.pyに適切な引数を渡して実行するためのシェルスクリプトです。
            - get_architecture_args.py:
                - コマンドライン引数を解釈し、使用するモデルアーキテクチャのパラメータを取得するヘルパースクリプトです。
    - automata/: オートマトン（言語を認識するための抽象的な計算モデル）に関連する実装が含まれています。
        1. 計算の核：「 (Semiring)」による抽象化
        
        このディレクトリの最も重要な設計思想はSemiring（半環）による計算の汎用化です。オートマトン内の経路計算を、目的（最短経路、経路数、受理可能性など）に応じて差し替え可能にするための
        仕組みです。
        
        - [semiring.py](http://semiring.py/):
            - Semiringの抽象基底クラス（インターフェース）を定義します。すべてのセiringはこのクラスを継承します。
        - boolean_semiring.py:
            - ある文字列が言語に受理されるか否か（True/False）を計算するためのセiringです。
        - tropical_semiring.py:
            - 経路の重み（コスト）の最小値を計算するためのセiringです（最短経路問題）。
        - counting_semiring.py:
            - ある文字列を受理する経路の総数を数え上げるためのセiringです。
        - log_counting_semiring.py:
            - counting_semiringと同様に経路数を数えますが、巨大な数になってもオーバーフローしないよう、対数領域で計算を行います。数値的な安定性のために重要です。
        1. オートマトン本体とアルゴリズム
        
        上記セiringを実際に利用して計算を行う、オートマトン本体と関連アルゴリズムです。
        
        - [automaton.py](http://automaton.py/):
            - オートマトン（状態、遷移、アルファベットなど）の基本的なデータ構造を定義する、中心的なクラスです。
        - finite_automaton.py:
            - automaton.pyを継承した、具体的な有限オートマトン（FA）の実装です。
        - finite_automaton_allsum.py:
            - 入力文字列に対して、指定されたSemiringを用いてすべての可能な経路にわたる計算（all-path-sum）を実行するアルゴリズムです。このファイルが、オートマトンとセiringを結びつける「計算エンジン」の役割を果たします。
        - fixed_point_iteration.py:
            - 不動点反復法を実装しています。これは、オートマトンの状態 reachable（到達可能）かどうかを判断するなど、様々な解析アルゴリズムの基礎となります。
        - create_state_annotator_fst.py:
            - 非常に興味深いファイルです。これは、文字列を入力として受け取り、その文字列を処理する際にオートマトンが通過する状態の系列を注釈（annotate）として出力する有限状態トランスデュ
            ーサ（FST）を作成します。これは、ニューラルネットワークに「正解の計算過程」を教えるための教師データを生成するのに使われる可能性があります。
        - [lehmann.py](http://lehmann.py/):
            - Lehmannに関連する特定のオートマトンアルゴリズムの実装だと思われます。
        - [reserved.py](http://reserved.py/):
            - ε（空文字）や未知の記号など、システム内で特別な意味を持つ予約済み記号を定義しています。
        
        まとめ
        
        このディレクトリは、単に文字列を受理するだけでなく、「どのように受理するか」を柔軟に計算できる、高度に抽象化されたオートマトンライブラリです。セiringを切り替えるだけで、同じオートマトンを使って最短経路、経路数、受理可能性などを自在に計算できる、非常に強力なツールキットと言えます。
        
    - hand_picked_languages/: dyck_k_m（カッコの対応）や
    binary_addition（2進数の足し算）など、特定の規則を持つ言語を手動で定義したコードが含まれています。これらがモデルの訓練・評価データになります。
        1. 算術・計数能力をテストする言語
        - binary_addition.py: 2進数の足し算 (例: "101+11=1000")。モデルが算術規則を学習できるかをテストします。
        - binary_multiplication.py: 2進数の掛け算。足し算より複雑な算術能力を要求します。
        - [parity.py](http://parity.py/): 偶奇性 (例: "1101" → 1が奇数個)。基本的な計数能力をテストします。
        - [majority.py](http://majority.py/): 過半数 (例: "aab" → aが過半数)。シンボルの出現頻度を比較する能力をテストします。
        - modular_arithmetic_simple.py: 単純な剰余演算。
        1. 記憶・構造認識能力をテストする言語
        - dyck_k_m.py: ダイク言語 (例:
        "([{}])")。入れ子構造を持つカッコの対応関係を認識する能力をテストします。これは文脈自由文法の典型例であり、単純な記憶以上の能力（スタックのような機能）が要求されます。
        - unmarked_reversal.py: 文字列の反転 (例: "abc" → "abccba")。入力シーケンスを記憶し、逆順に出力する能力をテストします。
        - marked_reversal.py: 例: "abc#cba" のように、中央に区切り文字がある反転。unmarkedよりも簡単な記憶タスクです。
        - cycle_navigation.py: サイクルの巡回。状態を記憶し、特定の順序で遷移する能力をテストします。
        - stack_manipulation.py: より直接的にスタック操作（PUSH, POP）を模倣した言語です。
        1. 並べ替え・ソート能力をテストする言語
        - odds_first.py: 奇数を先に (例: "1234" → "1324")。入力シーケンスをルールに基づいて並べ替える能力をテストします。
        - bucket_sort.py: バケットソート。より複雑なソートアルゴリズムの学習能力をテストします。
        
        補助的なファイルの役割
        
        - rayuela_util.py: このディレクトリでPythonコードとして定義された各言語を、プロジェクトの共通ライブラリである rayuela
        のオートマトン形式に変換するためのユーティリティ関数です。これが「接着剤」の役割を果たし、手書きのルールを形式的なオブジェクトに変換します。
        - save_automaton.py: rayuela_util.py
        を使って生成したオートマトンを、ファイルとして保存するためのスクリプトです。一度保存しておけば、データセット生成の際に毎回言語を定義し直す必要がなくなり、効率的です。
        - binary_util.py: 2進数関連の言語（加算、乗算）で共通して使われる処理をまとめたユーティリティです。
    - analysis/: experiments/analysisのスクリプトから呼び出される、実際の分析処理（プロット作成やテーブル生成）を行うPythonスクリプトが含まれています。
        1. プロット（図）作成
            - plot_cross_entropy_vs_edit_distance.py:
                - モデルの予測誤差（クロスエントロピー）と、入力文字列の「編集距離」（ノイズの量）との関係をプロットします。モデルがどの程度ノイズに頑健か（ロバストか）を分析するために使われ
                ます。
            - plot_cross_entropy_vs_length.py:
                - 予測誤差と入力文字列の「長さ」との関係をプロットします。モデルが長いシーケンスを苦手としていないかを分析します。
            - plot_num_edits_histogram.py:
                - 評価データセットに含まれる文字列の編集距離の分布をヒストグラムとしてプロットします。
            - plot_util.py:
                - 上記のプロット作成スクリプト群で共通して使われる補助関数（グラフのスタイル設定、軸ラベルの定義、ファイルの保存など）をまとめたユーティリティファイルです。
        2. テーブル（表）作成
        - print_main_table.py:
            - 実験結果の最も重要な部分をまとめたメインテーブル（論文の中心となる結果表など）を生成します。
        - print_table.py:
            - より汎用的なテーブル生成スクリプトです。
        - print_table_util.py:
            - テーブル作成のための補助関数（CSVの読み込み、数値のフォーマット、LaTeX形式への変換など）をまとめたユーティリティファイルです。
        - print_best_model.py:
            - 各言語タスクにおいて、最も性能が良かったモデルのパラメータやスコアを特定し、表示します。
        1. 詳細なエラー分析
        - sort_examples_by_difficulty.py:
            - モデルの評価結果に基づき、モデルが最も苦手とした（＝予測誤差が大きかった）サンプルを難しい順に並べ替えて出力します。これは、モデルがどのような種類の入力で間違いやすいのかを
            詳細に分析する「エラー分析」のために非常に重要です。
- `rayuela/`:
    - オートマトンや文法など、形式言語理論に関連する基本的なデータ構造やアルゴリズムを提供するライブラリです。recognizers/ のコードはこのライブラリに依存していると考えられます。
        
        主なファイルの役割
        
        1. 形式言語の基本要素 (The Core Building Blocks)
        - [symbol.py](http://symbol.py/):
            - 言語を構成する最小単位である「シンボル（記号）」を表現します。'a', 'b', '1' など、一つ一つの文字や記号がこれにあたります。
        - [alphabet.py](http://alphabet.py/):
            - シンボルの集合である「アルファベット」を定義します。例えば、{'a', 'b', 'c'} のような集合です。
        - [string.py](http://string.py/):
            - シンボルの列である「文字列（ストリング）」を表現します。
        - [state.py](http://state.py/):
            - オートマトンや文法における「状態（ステート）」を表現します。計算モデルのノードに相当します。
        1. 計算の抽象化 (The Core Abstraction)
        - [semiring.py](http://semiring.py/):
            - このライブラリの設計思想の中核をなす「Semiring（半環）」の抽象基底クラスです。これにより、オートマトンの経路計算を「受理可能か」「最短経路は」「経路数は」といった異なる目的に応じて、アルゴリズム本体を変更することなく差し替え可能にしています。これは recognizers/automata でも利用されている非常に重要な概念です。
        1. 補助的なデータ構造とアルゴリズム (Supporting Data Structures & Algorithms)
        - [partitions.py](http://partitions.py/):
            - 集合の「分割」を扱うためのデータ構造です。オートマトンの状態を等価なグループに分ける（状態最小化アルゴリズムなど）際に不可欠です。
        - [unionfind.py](http://unionfind.py/):
            - Union-Find（素集合データ構造）の実装です。要素を互いに素な集合に分類し、管理するために使われます。これも partitions
            と同様に、等価性の判定やグループ分けに非常に効率的なデータ構造です。
        - [datastructures.py](http://datastructures.py/):
            - ライブラリ内で共通して使われる、より一般的なデータ構造（特殊なキューやスタックなど）が定義されている可能性があります。
        - [misc.py](http://misc.py/), [universal.py](http://universal.py/):
            - 特定のカテゴリに分類しにくい、様々な補助関数や定数がまとめられています。
        1. 少し高度な概念
        - [termdep.py](http://termdep.py/):
            - "Term Dependency" の略だと思われます。これは、シンボル間の依存関係や、より複雑な項書き換えシステム（Term Rewriting
            System）に関連する概念を扱うためのコンポーネントである可能性があります。
    - fsa/: 有限状態オートマトン (Finite State Automaton) の実装。
        1. 基本的なデータ構造 (Core Data Structures)
            - [fsa.py](http://fsa.py/): 有限状態オートマトン（FSA）のクラスを定義する、このディレクトリで最も中心的なファイルです。FSAは文字列を「受理」または「拒絶」するモデルです。
            - [fst.py](http://fst.py/): 有限状態トランスデューサ（FST）のクラスを定義します。FSTは、入力文字列を出力文字列に「変換」するモデルで、翻訳やアノテーションなど、より高度なタスクに使われます。
            - [arc.py](http://arc.py/): オートマトンの状態と状態を結ぶ「アーク（遷移）」を表現するクラスです。アークは通常、入力シンボル、出力シンボル（FSTの場合）、および重みを持ちます。
            - fsa_classes.py: FSA/FSTに関連する、その他の補助的なクラスがまとめられている可能性があります。
        2. 計算エンジンと中核アルゴリズム (Computational Engine & Core Algorithms)
        - [pathsum.py](http://pathsum.py/): オートマトンの計算エンジンです。与えられた入力文字列に対して、base/semiring.pyで定義されたセiring（半環）を用いて、すべての可能な経路の「合計（sum）」を計算します。
        セiringを切り替えることで、この「合計」が「最短経路のコスト」になったり、「経路の総数」になったりします。
        - [transformer.py](http://transformer.py/): FSA/FSTを変換（transform）するためのアルゴリズム集です。以下のような、形式言語理論における標準的な操作が含まれていると推測されます。
            - 決定化（Determinization）: 非決定性オートマトンを等価な決定性オートマトンに変換します。
            - 最小化（Minimization）: 状態数が最小の等価なオートマトンに変換します。
            - 和（Union）、積（Intersection）、連結（Concatenation）: 複数のオートマトンを組み合わせて新しいオートマトンを作ります。
        - [scc.py](http://scc.py/): 強連結成分（Strongly Connected Components）を求めるアルゴリズムです。オートマトン内のサイクル構造を解析するために使われます。
        1. オートマトンの学習 (Automata Learning)
        このライブラリの非常に高度な機能です。与えられたサンプル（文字列の集合）から、その言語を生成するオートマトンを自動的に推論（学習）するアルゴリズムが含まれています。
        - [learning.py](http://learning.py/): 学習アルゴリズムの汎用的なインターフェースや基盤を提供します。
        - [angluin.py](http://angluin.py/): 正則言語（FSAが認識する言語）の学習アルゴリズムとして非常に有名な「L\*（エルスター）アルゴリズム」の実装です。
        - [beimel.py](http://beimel.py/), [hankel.py](http://hankel.py/): L\*アルゴリズム以外にも、オートマトンの学習や性質の検証に関連する、より専門的なアルゴリズムや数学的な道具（ハンケル行列など）が実装されています。
        1. 生成とサンプリング (Generation and Sampling)
        - [generator.py](http://generator.py/): ランダムなFSAを生成します。アルゴリズムのテストなどに使われます。
        - [random.py](http://random.py/), [sampler.py](http://sampler.py/): 与えられたFSAが受理する言語から、ランダムに文字列をサンプリング（生成）します。これは、ニューラルネットワークの訓練データを生成する際に不可欠です。
        1. その他
        - [examples.py](http://examples.py/): テストやデモ用の、具体的なFSAのサンプルが定義されています。
        - [utils.py](http://utils.py/): 上記のいずれにも分類されない、FSA関連の補助的な関数がまとめられています。
    - cfg/: 文脈自由文法 (Context-Free Grammar) の実装。
        
        主な役割とファイル分類
        
        1. 基本的なデータ構造 (Core Data Structures)
        - [cfg.py](http://cfg.py/): 文脈自由文法（CFG）のクラスを定義する、このディレクトリで最も中心的なファイルです。CFGは、非終端記号、終端記号、開始記号、生成規則の集合から構成されます。
        - [production.py](http://production.py/): CFGの「生成規則（プロダクション）」を表現するクラスです。例えば、S -> NP VP のような規則を定義します。
        - [nonterminal.py](http://nonterminal.py/): CFGの「非終端記号」を表現するクラスです。S (文), NP (名詞句), VP (動詞句) などがこれにあたります。
        - labeled_cfg.py: 通常のCFGに加えて、規則や非終端記号にラベル（追加情報）を付与できるCFGの実装です。構文解析結果に意味的な情報を付加する際に有用です。
        1. 構文解析と関連アルゴリズム (Parsing and Related Algorithms)
        - [parser.py](http://parser.py/): 入力文字列が与えられたCFGによって生成可能かどうかを判断し、その構文構造（パースツリー）を構築する「構文解析器（パーサー）」の実装です。
        - [pda.py](http://pda.py/): プッシュダウンオートマトン（Pushdown Automaton, PDA）の実装です。CFGと等価な計算能力を持つ抽象機械であり、CFGの構文解析の理論的基盤となります。
        - [transformer.py](http://transformer.py/): CFGを変換（transform）するためのアルゴリズム集です。以下のような操作が含まれると推測されます。
            - チョムスキー標準形（Chomsky Normal Form, CNF）への変換: 構文解析アルゴリズム（CYK法など）で扱いやすい形に変換します。
            - 左再帰の除去: 再帰的な規則を変換し、特定のパーサーが無限ループに陥るのを防ぎます。
        - [prefix.py](http://prefix.py/): 構文解析における接頭辞（prefix）に関連する概念やアルゴリズムを扱うファイルです。
        1. 特殊な文法と概念 (Specialized Grammars and Concepts)
        - bilexical_grammar.py: バイレキシカル文法の実装です。単語ペア間の依存関係を直接モデル化する文法で、依存構造解析（Dependency Parsing）に関連します。
        - [dependency.py](http://dependency.py/): 依存構造に関連する概念やアルゴリズムを扱うファイルです。
        - [brzozowski.py](http://brzozowski.py/): Brzozowskiのアルゴリズムに関連する実装です。これは、正規表現の導関数（derivative）を用いた正規表現マッチングや、オートマトンの最小化などに使われることがあります
        。CFGの文脈で使われる場合は、文法変換や解析の効率化に関連する可能性があります。
        - [treesum.py](http://treesum.py/): パースツリーの「合計（sum）」を計算するアルゴリズムです。fsa/pathsum.pyと同様に、セiringを用いて、異なる目的（最も確率の高いツリー、ツリーの総数など）でツリーを評価
        するために使われる可能性があります。
        1. 生成とサンプリング (Generation and Sampling)
        - [random.py](http://random.py/), [sampler.py](http://sampler.py/):
        与えられたCFGが生成する言語から、ランダムに文字列やパースツリーをサンプリング（生成）します。これは、ニューラルネットワークの訓練データを生成する際に利用されます。
        1. ユーティリティと補助 (Utility and Support)
        - [examples.py](http://examples.py/): テストやデモ用の、具体的なCFGのサンプルが定義されています。
        - [exceptions.py](http://exceptions.py/): CFGの操作中に発生する可能性のある例外を定義しています。
        - [misc.py](http://misc.py/): 上記のいずれにも分類されない、CFG関連の補助的な関数がまとめられています。
- `data/`:
    - ソースコード以外のデータを格納します。
    - figures/: 論文やレポートに掲載するための図や表を生成するためのLaTeXファイル (.tex) が含まれています。実験結果を整形して出力するために使われます。

# tests

# root