# NN-Recognizers

このリポジトリは、形式言語を認識するニューラルネットワークの訓練と評価を行うためのコードです。実験を再現するために必要なすべてのコードと、開発環境を複製するためのDockerイメージ定義が含まれています。

## ディレクトリ構成

*   `experiments/`: すべての実験と図を再現するための高レベルなスクリプトが含まれています。
*   `scripts/`: ソフトウェア環境のセットアップ、コンテナイメージのビルド、コンテナの実行、Pythonパッケージのインストールなどのための補助スクリプトが含まれています。
*   `src/`: ニューラルネットワークの訓練やデータ生成などのためのソースコードが含まれています。
    *   `recognizers/`: ニューラルネットワークの訓練、データ生成などを行うソースコードです。
        *   `analysis/`: プロット生成、予測分析などのためのコードです。
        *   `automata/`: オートマトンのためのデータ構造とアルゴリズムです。
        *   `hand_picked_languages/`: 各言語の実装です。
        *   `neural_networks/`: ニューラルネットワークの訓練と評価のためのコードです。
        *   `string_sampling/`: 正例と負例の文字列をサンプリングするためのコードです。
    *   `rayuela/`: オートマトンを扱うための補助ライブラリです。
*   `tests/`: `src/`以下のコードに対するpytestユニットテストが含まれています。

## インストールとセットアップ

再現性を確保するため、このコードは[`Dockerfile-dev`](Dockerfile-dev)で定義された[Docker](https://www.docker.com/)コンテナ内で開発・実行されました。このコードを実行するには、自分でDockerイメージをビルドして実行するか、`Dockerfile-dev`を参考に自身のシステムにソフトウェア環境をセットアップします。また、同等の[Singularity](https://sylabs.io/docs/#singularity)イメージをビルドして、Dockerが利用できないHPCクラスタなどで使用することも可能です。

### Dockerの使用

Dockerイメージを使用するには、まず[Dockerをインストール](https://www.docker.com/get-started)する必要があります。GPUで実験を実行する場合は、NVIDIAドライバが正しくセットアップされ、[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)がインストールされていることを確認してください。

Dockerイメージのビルド、コンテナの起動、そしてコンテナ内でのbashシェルの起動を自動的に行うには、次のコマンドを実行します。

    $ bash scripts/docker_shell.bash --build

一度イメージをビルドすれば、再度ビルドする必要はありません。その後は、単に次のコマンドを実行します。

    $ bash scripts/docker_shell.bash

デフォルトでは、このスクリプトはGPUモードでコンテナを起動します。これはGPUがないマシンでは失敗します。CPUモードでのみ実行したい場合は、次のコマンドを使用します。

    $ bash scripts/docker_shell.bash --cpu

### Singularityの使用

Singularityは、共有計算環境により適した代替コンテナランタイムです（Apptainerとも呼ばれます）。

Singularityコンテナでコードを実行するには、まずDockerイメージを取得し、それをルートアクセス権のあるマシン（個人のPCやワークステーションなど）で`.sif`（Singularityイメージ）ファイルに変換する必要があります。これには、そのマシンにDockerと[Singularity](https://docs.sylabs.io/guides/latest/user-guide/quick_start.html)の両方をインストールする必要があります。上記の指示に従ってDockerイメージをビルド済みであれば、次のコマンドで`.sif`ファイルを作成できます。

    $ bash scripts/build_singularity_image.bash

これにより`neural-network-recognizers.sif`というファイルが作成されます。この処理には数分かかるのが正常です。その後、`.sif`ファイルをHPCクラスタにアップロードして使用できます。

Singularityコンテナ内でシェルを開くには、次のコマンドを使用します。

    $ bash scripts/singularity_shell.bash

これはNVIDIA GPUの有無にかかわらず動作しますが、GPUがない場合は警告が出力されます。

### 追加のセットアップ

どの方法（Docker、Singularity、またはコンテナなし）でコードを実行するかにかかわらず、*コンテナシェル内で*一度だけ次のスクリプトを実行する必要があります。

    $ bash scripts/setup.bash

このスクリプトは、コードが必要とするPythonパッケージをインストールします。パッケージはシステムワイドではなく、ローカルディレクトリに保存されます。

## コードの実行

`src/`以下のすべてのファイルは、Poetryが提供するPythonパッケージにアクセスできるよう、`poetry`を使用して実行する必要があります。つまり、すべてのコマンドの前に`poetry run`を付けるか、事前に`poetry shell`を実行してPoetryの仮想環境が有効なシェルに入ります。BashスクリプトがPythonスクリプトを呼び出す可能性があるため、PythonスクリプトとBashスクリプトの両方をPoetryで実行する必要があります。`src/`以下のすべてのBashスクリプトは、`src/`をカレントワーキングディレクトリとして実行してください。

`scripts/`と`experiments/`以下のすべてのスクリプトは、トップレベルディレクトリをカレントワーキングディレクトリとして実行してください。

## 実験の実行

[`experiments/`](experiments)ディレクトリには、すべての実験とプロットを再現するためのスクリプトが含まれています。一部のスクリプトは、計算クラスタにジョブを投入することを目的としています。これらはコンテナの外で実行する必要があります。ご自身の計算クラスタに合わせて[`experiments/submit_job.bash`](experiments/submit_job.bash)ファイルを編集する必要があります。他のスクリプトはプロットやテーブルの出力用であり、コンテナ内で実行する必要があります。

### データセット生成

データセットをゼロから生成するためのスクリプトは`experiments/dataset_generation/`にあります。すべてのデータセットは固定のランダムシードでサンプリングされるため、結果は決定論的です。

データセット生成は以下のステップで構成されます：

1.  (正則言語のみ) 言語のDFAを.ptファイルに書き出す。
2.  (正則言語のみ) サンプリングに使用できるよう、DFAに重みプッシュを実行する。
3.  各分割（訓練/検証/テスト）に対して正例と負例をランダムにサンプリングし、結果を平文ファイルとして保存する。
4.  平文ファイルを準備する。具体的には、すべての記号を整数に変換し、.ptファイルに保存する。

関連スクリプト：
*   `submit_prepare_automaton_language_jobs.bash`: すべての正則言語のデータセットを生成・準備します。
*   `submit_prepare_hand_coded_language_jobs.bash`: すべての非正則言語のデータセットを生成・準備します。
*   `submit_prepare_test_edit_distance_jobs.bash`: 編集距離のプロットに使用するデータセットを生成・準備します。`submit_prepare_automaton_language_jobs.bash`の後に実行する必要があります。

### ニューラルネットワークの訓練

関連スクリプトは`experiments/training/`にあります。データセットが生成・準備された後に実行してください。

*   `submit_train_and_evaluate_jobs.bash`: すべての言語で、すべてのモデルの訓練と評価を行います。
*   `submit_rerun_evaluate_jobs.bash`: モデルの再訓練なしに、評価のみを再実行します。モデルの再訓練を必要としない評価エラーが発生した場合に便利です。

### 分析

関連スクリプトは`experiments/analysis/`にあります。モデルが訓練・評価された後に実行してください。

*   `print_full_tables.bash`: すべての言語に対する省略なしの結果テーブルを生成します。
*   `print_main_tables.bash`: すべての帰納バイアスと表現力の実験をまとめたテーブル、および最良の損失関数を示すテーブルを生成します。
*   `plot_cross_entropy_vs_edit_distance.bash`: クロスエントロピー対編集距離のプロットを生成します。
*   `print_hardest_examples.bash`: 特定の言語とアーキテクチャについて、テストセットの例をクロスエントロピーが高い順（難しい順）にソートします。

## 実験のカスタマイズ

このフレームワークの実験は、`src/recognizers/neural_networks/train.py` スクリプトに渡すコマンドライン引数を変更することで、柔軟にカスタマイズできます。`experiments/training/submit_train_and_evaluate_jobs.bash` のようなジョブ提出スクリプト内でこれらの引数を設定することで、独自の実験を実行できます。

### データセットのカスタマイズ

データセットは `src/recognizers/neural_networks/prepare_data.py` を用いて生成されます。主要な引数は以下の通りです。

*   `--language`: `src/recognizers/hand_picked_languages` で定義されている言語の名前（例: `parity`, `binary_addition`）。
*   `--num-train-examples`, `--num-dev-examples`, `--num-test-examples`: 各データ分割のサンプル数を指定します。
*   `--max-length`: 生成される文字列の最大長を指定します。

### モデルアーキテクチャとハイパーパラメータの選択

`train.py` は、モデルのアーキテクチャとハイパーパラメータを制御するための多くの引数を受け取ります。

**1. アーキテクチャの選択:**

`--architecture` 引数で使用するモデルのクラス名を指定します。利用可能なアーキテクチャは `src/recognizers/neural_networks/model_interface.py` で定義されている `get_architectures()` 関数で確認できます。一般的な選択肢は以下の通りです。

*   `Transformer`
*   `LSTM`
*   `Ngram`

**2. 共通のハイパーパラメータ:**

*   `--language`: 訓練に使用する言語。
*   `--epochs`: 訓練エポック数。
*   `--learning-rate`: 学習率。
*   `--batch-size`: バッチサイズ。
*   `--loss-function`: 損失関数（例: `cross_entropy`, `length_normalized_cross_entropy`）。

**3. アーキテクチャ固有のハイパーパラメータ:**

選択したアーキテクチャに応じて、追加の引数を指定できます。例えば、`--architecture Transformer` を選択した場合、以下のような引数が利用可能です。

*   `--num-layers`: Transformerのエンコーダ層の数。
*   `--num-heads`: Multi-head attentionのヘッド数。
*   `--embedding-dim`: 埋め込みベクトルの次元数。
*   `--hidden-dim`: Feed-forwardネットワークの隠れ層の次元数。

これらの引数は、`ModelInterface` を継承した各モデルクラスの `add_architecture_args` 静的メソッドで定義されています。

### カスタム実験の実行例

例えば、Dyck-2言語 (`dyck_2_10`) に対して、4層で8ヘッドのTransformerモデルを、学習率0.0001、バッチサイズ64で50エポック訓練する場合、`submit_train_and_evaluate_jobs.bash` のようなスクリプト内で、`train.py` を以下のような引数で呼び出すことになります。

```bash
poetry run python src/recognizers/neural_networks/train.py \
    --language dyck_2_10 \
    --architecture Transformer \
    --epochs 50 \
    --learning-rate 0.0001 \
    --batch-size 64 \
    --num-layers 4 \
    --num-heads 8 \
    --embedding-dim 256 \
    --hidden-dim 1024
```