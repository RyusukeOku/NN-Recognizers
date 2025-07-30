import json
import pathlib
import pandas as pd

def load_json_from_file(file_path):
    if file_path.exists() and file_path.is_file():
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"警告: {file_path} のJSONデコードに失敗しました。")
            return None
        except Exception as e:
            print(f"警告: {file_path} の読み込み中にエラーが発生しました: {e}")
            return None
    return None

def collect_results(base_models_dir):
    results = []
    models_path = pathlib.Path(base_models_dir)

    if not models_path.exists() or not models_path.is_dir():
        print(f"エラー: ベースモデルディレクトリ '{base_models_dir}' が見つかりません。")
        return pd.DataFrame()
    
    for language_dir in models_path.iterdir():
        if not language_dir.is_dir():
            continue
        for arch_dir in language_dir.iterdir():
            if not arch_dir.is_dir():
                continue
            for loss_dir in arch_dir.iterdir():
                if not loss_dir.is_dir():
                    continue
                for val_dir in loss_dir.iterdir():
                    if not val_dir.is_dir():
                        continue
                    for trial_dir in val_dir.iterdir():
                        if not trial_dir.is_dir():
                            continue
                        
                        eval_path = trial_dir / "eval"
                        if not eval_path.exists() or not eval_path.is_dir():
                            continue

                        training_json_path = eval_path / "training.json"
                        test_json_path = eval_path / "test.json"

                        training_data = load_json_from_file(training_json_path)
                        test_data = load_json_from_file(test_json_path)

                        result_entry = {
                            "language": language_dir.name,
                            "architecture": arch_dir.name,
                            "loss_terms": loss_dir.name,
                            "validation_set": val_dir.name,
                            "trial": trial_dir.name,
                            "train_accuracy": None,
                            "train_loss": None,
                            "test_accuracy": None,
                            "test_loss": None,
                        }

                        if training_data and "scores" in training_data:
                            result_entry["train_accuracy"] = training_data["scores"].get("recognition_accuracy")
                            result_entry["train_loss"] = training_data["scores"].get("recognition_cross_entropy")
                        
                        if test_data and "scores" in test_data:
                            result_entry["test_accuracy"] = test_data["scores"].get("recognition_accuracy")
                            result_entry["test_loss"] = test_data["scores"].get("recognition_cross_entropy")
                        
                        if training_data or test_data:
                            results.append(result_entry)

    if not results:
        print("結果が見つかりませんでした。")
        return pd.DataFrame()
        
    return pd.DataFrame(results)

def main():
    base_models_directory = "data/models"
    mean_output_csv_path = "results/eval_results_mean_summary_annotation_with_F.csv"
    
    df_results = collect_results(base_models_directory)

    if not df_results.empty:
        numeric_cols = ['train_accuracy', 'train_loss', 'test_accuracy', 'test_loss']
        for col in numeric_cols:
            if col in df_results.columns:
                df_results[col] = pd.to_numeric(df_results[col], errors='coerce')

        grouping_keys = ['language', 'architecture', 'loss_terms', 'validation_set']
        existing_grouping_keys = [key for key in grouping_keys if key in df_results.columns]

        if not existing_grouping_keys:
            print("エラー: グループ化するためのキー列が見つかりません。")
            return

        aggregation_functions = {
            'train_accuracy': 'mean',
            'train_loss': 'mean',
            'test_accuracy': 'mean',
            'test_loss': 'mean',
            'trial': 'count'
        }
        
        existing_aggregation_functions = {k: v for k, v in aggregation_functions.items() if k in df_results.columns or k == 'trial'}
        
        df_mean = df_results.groupby(existing_grouping_keys).agg(existing_aggregation_functions).reset_index()
        
        if 'trial' in df_mean.columns:
            df_mean.rename(columns={'trial': 'num_trials'}, inplace=True)
        
        print("\n--- 実験結果の平均値 ---")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
            print(df_mean)

        try:
            output_path = pathlib.Path(mean_output_csv_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df_mean.to_csv(output_path, index=False)
            print(f"\n平均値を {output_path} に保存しました。")
        except Exception as e:
            print(f"エラー: 平均値をCSVファイルに保存できませんでした: {e}")
    else:
        print("表示する結果がありません。")

if __name__ == "__main__":
    main()