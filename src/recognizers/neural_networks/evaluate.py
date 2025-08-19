import argparse
import collections
import json
import math
import pathlib
import sys

import torch

from rau.tasks.common.training_loop import MicroAveragedScoreAccumulator

from recognizers.tools.jsonl import write_json_line
from recognizers.neural_networks.data import load_prepared_data_from_directory
from recognizers.neural_networks.model_interface import RecognitionModelInterface
from recognizers.neural_networks.training_loop import generate_batches, get_loss_terms
from recognizers.automata.format_checker import check_string_format

def evaluate(model, model_interface, batches, num_examples, language_name: str = None):
    device = model_interface.get_device(None)
    example_scores = [None] * num_examples
    model.eval()

    with torch.inference_mode():
        for indexed_batch in batches:
            batch_to_process = indexed_batch
            accepted_indexed_batch = []

            if language_name:
                for i, (x, (original_index, (true_label, _))) in enumerate(indexed_batch):
                    if check_string_format(language_name, x):
                        accepted_indexed_batch.append(indexed_batch[i])
                    else:
                        # 形式不一致の場合、NNが「不受理」と判断したのと同じ扱いにする
                        predicted_label_is_accepted = False
                        is_correct = (true_label == predicted_label_is_accepted)
                        
                        accuracy_numerator = 1.0 if is_correct else 0.0
                        loss_numerator = 0.0 if is_correct else math.inf

                        example_scores[original_index] = {
                            'loss': (loss_numerator, 1),
                            'accuracy': (accuracy_numerator, 1)
                        }
                
                if not accepted_indexed_batch:
                    continue
                batch_to_process = accepted_indexed_batch

            batch = [(x, d) for x, (i, d) in batch_to_process]
            prepared_batch = model_interface.prepare_batch(batch, device)

            batch_score_dict = get_loss_terms(
                model,
                model_interface,
                prepared_batch,
                numerator_reduction='none',
                denominator_reduction='none',
                label_smoothing_factor=0.0,
                include_accuracy=True
            )
            
            example_score_dicts = split_score_dict(batch, batch_score_dict)
            
            for (x, (i, d)), example_score_dict in zip(batch_to_process, example_score_dicts):
                example_scores[i] = example_score_dict

    return example_scores

class DictScoreAccumulator:

    def __init__(self):
        super().__init__()
        self.scores = collections.defaultdict(MicroAveragedScoreAccumulator)

    def update(self, scores: dict[str, tuple[float, float]]) -> None:
        for key, (numerator, denominator) in scores.items():
            self.scores[key].update(numerator, denominator)

    def get_value(self) -> dict[str, float]:
        return { k : v.get_value() for k, v in self.scores.items() }

def split_score_dict(batch, batch_score_dict):
    batch_score_dict = {
        k : (n.tolist(), d.tolist() if d is not None else d)
        for k, (n, d) in batch_score_dict.items()
    }
    positive_index = 0
    for index, example in enumerate(batch):
        label = example[1][0]
        example_score_dict = {}
        for key, (numerator, denominator) in batch_score_dict.items():
            if len(numerator) < len(batch):
                if label:
                    example_score_dict[key] = (
                        numerator[positive_index],
                        denominator[positive_index] if denominator is not None else 1
                    )
            else:
                example_score_dict[key] = (
                    numerator[index],
                    denominator[index] if denominator is not None else 1
                )
        yield example_score_dict
        positive_index += int(label)

def main():

    model_interface = RecognitionModelInterface(
        use_load=True,
        use_init=False,
        use_output=False,
        require_output=False
    )

    parser = argparse.ArgumentParser(
        description=
        'Evaluate a language model on a dataset. Output the results as JSON.'
    )
    parser.add_argument('--training-data', type=pathlib.Path, required=True,
        help='A directory containing training data. The file '
             '<training-data>/datasets/<input>/main.prepared will be used as '
             'input, and the file '
             '<training-data>/main.vocab will be used as the vocabulary.')
    parser.add_argument('--datasets', nargs='+', required=True,
        help='Names of datasets in the training data directory that will be '
             'used as input. The file '
             '<training-data>/datasets/<dataset>/main.prepared will be used as '
             'input. Multiple datasets can be passed. The name "training" '
             'can be used to evaluate on the training data.')
    parser.add_argument('--output', type=pathlib.Path, required=True,
        help='A directory where output files will be written.')
    parser.add_argument('--batching-max-tokens', type=int, required=True,
        help='The maximum number of tokens allowed per batch.')
    model_interface.add_arguments(parser)
    model_interface.add_forward_arguments(parser)
    parser.add_argument('--use-format-filter', action='store_true',
        help='Enable regex-based pre-filtering of input strings for evaluation.')
    args = parser.parse_args()

    saver = model_interface.construct_saver(args)

    language_name = args.training_data.name if args.use_format_filter else None
    if language_name:
        print(f'Format filter is enabled for language: {language_name}')

    for dataset in args.datasets:
        if dataset == 'training':
            input_directory = args.training_data
        else:
            input_directory = args.training_data / 'datasets' / dataset
        examples = load_prepared_data_from_directory(
            input_directory,
            model_interface
        )
        examples = [(x, (i, d)) for i, (x, d) in enumerate(examples)]
        batches = generate_batches(examples, args.batching_max_tokens)
        scores = evaluate(saver.model, model_interface, batches, len(examples), language_name=language_name)
        accumulator = DictScoreAccumulator()
        example_scores_path = args.output / f'{dataset}.jsonl'
        print(f'writing {example_scores_path}')
        with example_scores_path.open('w') as fout:
            for score_dict in scores:
                write_json_line(score_dict, fout)
                accumulator.update(score_dict)
        total_scores = accumulator.get_value()
        total_scores_path = args.output / f'{dataset}.json'
        print(f'writing {total_scores_path}')
        with total_scores_path.open('w') as fout:
            json.dump(dict(scores=total_scores), fout, indent=2)
        json.dump(total_scores, sys.stdout, indent=2)
        print()

if __name__ == '__main__':
    main()