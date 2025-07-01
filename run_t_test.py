import json
import pathlib
import pandas as pd
from scipy import stats
from itertools import product

def collect_accuracies(base_dir: str, model_type: str):
    """
    指定されたベースディレクトリからテストデータの正解率を収集する。
    """
    results = []
    base_path = pathlib.Path(base_dir)
    if not base_path.exists():
        print(f"警告: ディレクトリが見つかりません: {base_path}")
        return results

    # すべての test.json ファイルを探索
    for test_json_path in base_path.glob('**/eval/test.json'):
        try:
            # パスから実験設定の情報を抽出
            parts = test_json_path.parts
            trial = parts[-3]
            validation_data = parts[-4]
            loss_terms = parts[-5]
            architecture = parts[-6]
            language = parts[-7]

            with open(test_json_path, 'r') as f:
                data = json.load(f)
            
            accuracy = data.get('scores', {}).get('recognition_accuracy')

            if accuracy is not None:
                results.append({
                    'model_type': model_type,
                    'language': language,
                    'architecture': architecture,
                    'loss_terms': loss_terms,
                    'validation_data': validation_data,
                    'trial': trial,
                    'test_accuracy': accuracy
                })
        except (IndexError, FileNotFoundError, json.JSONDecodeError) as e:
            print(f"警告: ファイルの処理中にエラーが発生しました {test_json_path}: {e}")
            continue
            
    return results

def main(output_csv_path="t-test_results.csv"):
    """
    モデルの結果を収集し、t検定を実行して結果をCSVに保存する。
    """
    # 1. 両方のモデルグループからデータを収集
    original_results = collect_accuracies("data/models_original", "original")
    new_model_results = collect_accuracies("data/models", "new_model")

    if not original_results or not new_model_results:
        print("エラー: 片方または両方のモデルグループからデータを収集できませんでした。処理を中断します。")
        return

    all_results_df = pd.concat([pd.DataFrame(original_results), pd.DataFrame(new_model_results)], ignore_index=True)

    # 2. 実験設定ごとにグループ化
    grouped = all_results_df.groupby(['language', 'architecture', 'loss_terms', 'validation_data'])

    t_test_results = []

    # 3. 各グループでt検定を実行
    for name, group in grouped:
        original_accuracies = group[group['model_type'] == 'original']['test_accuracy']
        new_model_accuracies = group[group['model_type'] == 'new_model']['test_accuracy']

        # 両方のグループに十分なデータがあるか確認 (t検定には少なくとも2つのサンプルが必要)
        if len(original_accuracies) < 2 or len(new_model_accuracies) < 2:
            print(f"情報: {name} のデータが不足しているため、t検定をスキップします。")
            continue

        # 独立2標本t検定
        # equal_var=False を指定して Welch's t-test を実行 (2つのグループの分散が等しいと仮定しない)
        t_stat, p_value = stats.ttest_ind(new_model_accuracies, original_accuracies, equal_var=False, nan_policy='omit')

        t_test_results.append({
            'language': name[0],
            'architecture': name[1],
            'loss_terms': name[2],
            'validation_data': name[3],
            't_statistic': t_stat,
            'p_value': p_value,
            'mean_accuracy_original': original_accuracies.mean(),
            'mean_accuracy_new_model': new_model_accuracies.mean(),
            'n_original': len(original_accuracies),
            'n_new_model': len(new_model_accuracies)
        })

    if not t_test_results:
        print("t検定を実行できるデータがありませんでした。")
        return

    # 4. 結果をDataFrameに変換してCSVに保存
    results_df = pd.DataFrame(t_test_results)
    results_df = results_df.sort_values(by='p_value', ascending=True)

    try:
        results_df.to_csv(output_csv_path, index=False, float_format='%.6f')
        print(f"\nt検定の結果を {output_csv_path} に保存しました。")
        print("\n--- 検定結果サマリー ---")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 120):
             print(results_df)
    except Exception as e:
        print(f"エラー: 結果をCSVファイルに保存できませんでした: {e}")

if __name__ == "__main__":
    # 出力ファイル名を指定して実行
    main(output_csv_path="t-test_accuracy_comparison.csv")