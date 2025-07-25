import argparse
import os
import random
import torch
from pathlib import Path

from rayuela.fsa.fsa import FSA
from rayuela.fsa.fst import FST
from rayuela.base.semiring import Tropical, Real

def read_data(path: str) -> list[tuple[list[str], bool]]:
    """Reads data from a file, returning a list of (tokens, label) tuples."""
    data = []
    with open(path, "r") as f:
        for line in f:
            label, text = line.strip().split("\t", 1)
            data.append((text.split(), label == "+"))
    return data

def write_data(path: str, data: list[tuple[list[str], bool]]):
    """Writes data to a file."""
    with open(path, "w") as f:
        for tokens, label in data:
            label_str = "+" if label else "-"
            f.write(f"{label_str}\t{' '.join(tokens)}\n")

def annotate_data(data: list[tuple[list[str], bool]], annotator_fst: FST) -> list[tuple[list[str], bool]]:
    """Annotates the input part of the data using the provided FST."""
    annotated_data = []
    for tokens, label in data:
        if not tokens:
            annotated_data.append((tokens, label))
            continue
        
        input_fsa = FSA.from_string(tokens, R=annotator_fst.R)
        
        try:
            composed_fst = annotator_fst.compose(input_fsa)
            best_path = composed_fst.shortest_path()
            
            if best_path is None:
                annotated_tokens = tokens
            else:
                annotated_tokens = best_path.output_string
        except Exception:
            annotated_tokens = tokens
            
        annotated_data.append((annotated_tokens, label))
    return annotated_data

def reconstruct_fst_from_data(fst_data: dict) -> FST:
    """Reconstructs an FST from a dictionary."""
    semiring_map = {
        'tropical': Tropical,
        'real': Real
    }
    semiring = semiring_map.get(fst_data['semiring_type'])
    if semiring is None:
        raise ValueError(f"Unsupported semiring type: {fst_data['semiring_type']}")

    fst = FST(R=semiring)
    for state in fst_data['states']:
        fst.add_state(state)
    
    if fst_data['initial_state'] is not None:
        fst.set_I(fst_data['initial_state'])
    
    for final_state in fst_data['final_states']:
        fst.add_F(final_state)
    
    for p, i, o, q, w_val in fst_data['arcs']:
        fst.add_arc(p, i, o, q, semiring(w_val))
        
    return fst

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=str, required=True)
    parser.add_argument("--dev-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-state-annotations", action="store_true", help="Enable FST-based state annotations.")
    parser.add_argument("--fst-annotator-path", type=Path, help="Path to the FST data file.")
    
    args = parser.parse_args()

    if args.use_state_annotations and not args.fst_annotator_path:
        raise ValueError("The --fst-annotator-path must be specified when --use-state-annotations is enabled.")

    os.makedirs(args.output_dir, exist_ok=True)

    annotator = None
    if args.use_state_annotations:
        print("Loading and reconstructing FST annotator...")
        fst_data = torch.load(args.fst_annotator_path)
        annotator = reconstruct_fst_from_data(fst_data)
        print("FST annotator ready.")

    # Process training data
    train_data = read_data(args.train_path)
    if annotator:
        print("Annotating training data...")
        train_data = annotate_data(train_data, annotator)
    train_output_path = os.path.join(args.output_dir, "train.txt")
    write_data(train_output_path, train_data)
    print(f"Wrote processed training data to {train_output_path}")

    # Process development data
    if args.dev_path:
        dev_data = read_data(args.dev_path)
        if annotator:
            print("Annotating development data...")
            dev_data = annotate_data(dev_data, annotator)
        dev_output_path = os.path.join(args.output_dir, "dev.txt")
        write_data(dev_output_path, dev_data)
        print(f"Wrote processed development data to {dev_output_path}")

if __name__ == "__main__":
    main()