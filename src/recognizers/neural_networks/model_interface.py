import dataclasses
from typing import Optional

import torch
import torch.nn as nn

from rau.models.common.shared_embeddings import get_shared_embeddings
from rau.models.rnn import LSTM, SimpleRNN
from rau.models.transformer.positional_encodings import (
    SinusoidalPositionalEncodingCacher
)
from rau.tasks.common.model import pad_sequences
from rau.tools.torch.embedding_layer import EmbeddingLayer
from rau.tools.torch.init import smart_init, uniform_fallback
from rau.tools.torch.model_interface import ModelInterface
from rau.tools.torch.layer import Layer
from rau.tools.torch.compose import Composable
from rau.models.transformer.input_layer import get_transformer_input_unidirectional
from rau.models.transformer.unidirectional_encoder import UnidirectionalTransformerEncoderLayers
from rau.unidirectional import (
    EmbeddingUnidirectional,
    DropoutUnidirectional,
    OutputUnidirectional,
    Unidirectional
)
from rau.tools.torch.saver import read_saver, construct_saver
from rayuela.fsa.fsa import FSA

from .vocabulary import get_vocabularies
from .resettable_positional_encoding import ResettablePositionalInputLayer
from .ngram_head import NgramHead
from .lba import NeuralLBA
from .FSA_integrated_input_layer import FSAIntegratedInputLayer
import recognizers.automata.structural_fsas as structural_fsas
import argparse
import json
from pathlib import Path
from .data import load_vocabulary_data
import random

@dataclasses.dataclass
class ModelInput:
    input_sequence: torch.Tensor
    last_index: torch.Tensor
    positive_mask: Optional[torch.Tensor]
    input_ids: torch.Tensor

class HybridCSGModel(nn.Module):
    """
    指定されたメインアーキテクチャ（Transformer, RNN, LSTM）と
    NeuralLBAを組み合わせた汎用ハイブリッドモデル
    """
    def __init__(self,
                 input_vocabulary_size,
                 embedding_size,
                 main_model,
                 main_model_output_size,
                 lba_hidden_size,
                 lba_n_steps,
                 device,
                 shared_embeddings=None):
        super().__init__()

        self.embedding = EmbeddingLayer(
            vocabulary_size=input_vocabulary_size,
            output_size=embedding_size,
            use_padding=False,
            shared_embeddings=shared_embeddings
        )
        
        self.main_model = main_model
        self.sub_network = NeuralLBA(
            input_size=embedding_size,
            hidden_size=lba_hidden_size,
            n_steps=lba_n_steps,
            device=device
        )
        
        self.classifier = nn.Linear(main_model_output_size + lba_hidden_size, 1)

    def forward(self, input_sequence, last_index, **kwargs):
        embedded = self.embedding(input_sequence)
        
        main_output_sequence = self.main_model(embedded, include_first=True)
        last_main_hidden = torch.gather(
            main_output_sequence,
            1,
            last_index[:, None, None].expand(-1, -1, main_output_sequence.size(2))
        ).squeeze(1)
        
        lba_output = self.sub_network(embedded)
        
        combined_output = torch.cat((last_main_hidden, lba_output), dim=1)
        
        logits = self.classifier(combined_output).squeeze(-1)

        return logits, None, None

class ForceIncludeFirstFalse(Unidirectional):
    def __init__(self, module: Unidirectional):
        super().__init__()
        self.module = module
        self._composable_tags = getattr(module, '_composable_tags', {})

    def forward(self, input_sequence: torch.Tensor, *args, **kwargs):
        kwargs['include_first'] = False
        return self.module(input_sequence, *args, **kwargs)

class RecognitionModelInterface(ModelInterface):

    def set_attributes_from_args(self, args):
        """
        Sets attributes on the interface from the parsed arguments.
        This is called during training, before data loading, to make
        argument-dependent information available to the data loading process.
        """
        self.architecture = args.architecture
        self.use_language_modeling_head = args.use_language_modeling_head
        self.use_next_symbols_head = args.use_next_symbols_head
        self.add_ngram_head_n = getattr(args, 'add_ngram_head_n', 0)
        
        self.uses_bos = args.architecture == 'transformer' or \
                        (args.architecture == 'hybrid_csg' and getattr(args, 'hybrid_base_architecture', None) == 'transformer')
        self.uses_eos = args.use_language_modeling_head or args.use_next_symbols_head

    def add_more_init_arguments(self, group):
        group.add_argument('--architecture', choices=['transformer', 'rnn', 'lstm', 'hybrid_csg'],
            help='The type of neural network architecture to use.')
        group.add_argument('--add-ngram-head-n', type=int, default=0,
            help='If greater than 0, add an n-gram head of size n to the model.')
        group.add_argument('--hybrid-base-architecture', choices=['transformer', 'rnn', 'lstm'],
            help='(hybrid_csg) The base architecture to combine with the LBA.')
        group.add_argument('--num-layers', type=int,
            help='(transformer, rnn, lstm) Number of layers.')
        group.add_argument('--d-model', type=int,
            help='(transformer) The size of the vector representations used in the transformer.')
        group.add_argument('--num-heads', type=int,
            help='(transformer) The number of attention heads used in each layer.')
        group.add_argument('--feedforward-size', type=int,
            help='(transformer) The size of the hidden layer of the feedforward network in each feedforward sublayer.')
        group.add_argument('--dropout', type=float,
            help='Dropout rate for all applicable layers.')
        group.add_argument('--hidden-units', type=int,
            help='(rnn, lstm) Number of hidden units to use in the hidden state.')
        group.add_argument('--init-scale', type=float,
            help='The scale used for the uniform distribution from which certain parameters are initialized.')
        group.add_argument('--use-language-modeling-head', action='store_true', default=False,
            help='Add a language modeling head to the model.')
        group.add_argument('--use-next-symbols-head', action='store_true', default=False,
            help='Add a next symbols prediction head to the model.')
        group.add_argument('--positional-encoding', choices=['sinusoidal', 'resettable'], default='sinusoidal',
            help='(transformer only) The type of positional encoding to use.')
        group.add_argument('--reset-symbols', type=str, default='+#-=×*()[]_POPUSH',
            help='(transformer with resettable encoding only) The symbols that reset the position.')
        group.add_argument('--embedding-size', type=int, help='(hybrid_csg) Embedding size.')
        group.add_argument('--lba-hidden-size', type=int, default=40, help='(hybrid_csg) Hidden size for NeuralLBA.')
        group.add_argument('--lba-n-steps', type=int, default=100, help='(hybrid_csg) Number of steps for NeuralLBA.')
        group.add_argument('--use-fsa-features', action='store_true', default=False,
            help='(transformer only) Use FSA state features in the input layer.')
        group.add_argument('--fsa-name', type=str,
            help='(transformer with --use-fsa-features) Name of the structural FSA to use.')
        group.add_argument('--fsa-embedding-dim', type=int,
            help='(transformer with --use-fsa-features) Dimension of the FSA state embeddings.')


    def get_kwargs(self, args, vocabulary_data):
        uses_bos = args.architecture == 'transformer' or (args.architecture == 'hybrid_csg' and args.hybrid_base_architecture == 'transformer')
        uses_output_vocab = args.use_language_modeling_head or args.use_next_symbols_head
        input_vocab, output_vocab = get_vocabularies(
            vocabulary_data,
            use_bos=uses_bos,
            use_eos=uses_output_vocab
        )
        reset_symbol_ids = None
        if args.positional_encoding == 'resettable':
            reset_symbol_ids = {input_vocab.to_int(s) for s in args.reset_symbols if s in input_vocab}
        
        kwargs = dict(
            architecture=args.architecture,
            add_ngram_head_n=getattr(args, 'add_ngram_head_n', 0),
            hybrid_base_architecture=getattr(args, 'hybrid_base_architecture', None),
            embedding_size=getattr(args, 'embedding_size', None),
            num_layers=args.num_layers,
            d_model=args.d_model,
            num_heads=args.num_heads,
            feedforward_size=args.feedforward_size,
            dropout=args.dropout,
            hidden_units=args.hidden_units,
            use_language_modeling_head=args.use_language_modeling_head,
            use_next_symbols_head=args.use_next_symbols_head,
            input_vocabulary_size=len(input_vocab),
            output_vocabulary_size=len(output_vocab) if uses_output_vocab else None,
            bos_index=input_vocab.bos_index if uses_bos else None,
            eos_index=output_vocab.eos_index if uses_output_vocab else None,
            positional_encoding=args.positional_encoding,
            reset_symbol_ids=reset_symbol_ids,
            lba_hidden_size=getattr(args, 'lba_hidden_size', 40),
            lba_n_steps=getattr(args, 'lba_n_steps', 50)
        )

        kwargs['use_fsa_features'] = args.use_fsa_features
        if args.use_fsa_features:
            if args.architecture != 'transformer':
                raise ValueError('--use-fsa-features is only supported for --architecture=transformer')
            if args.fsa_name is None:
                raise ValueError('--fsa-name is required when using --use-fsa-features')
            if args.fsa_embedding_dim is None:
                raise ValueError('--fsa-embedding-dim is required when using --use-fsa-features')

            fsa_func_name = f'{args.fsa_name}_structural_fsa_container'
            if not hasattr(structural_fsas, fsa_func_name):
                raise ValueError(f"Unknown FSA name: {args.fsa_name}")
            
            fsa_func = getattr(structural_fsas, fsa_func_name)
            fsa_container, fsa_alphabet = fsa_func()

            kwargs['fsa_name'] = args.fsa_name
            kwargs['fsa_container'] = fsa_container
            kwargs['fsa_alphabet'] = fsa_alphabet
            kwargs['fsa_embedding_dim'] = args.fsa_embedding_dim
            kwargs['word_vocab'] = input_vocab

        return kwargs

    def construct_saver(self, args, vocabulary_data=None):
        # Override the base method to correctly handle loading vs. training.
        device = self.get_device(args)
        # Check if we are loading a pre-trained model
        if self.use_load and getattr(args, 'load_model', None) is not None:
            # --- LOADING PATH ---
            if args.load_model is None:
                self.fail_argument_check('Argument --load-model is missing.')

            # 1. Manually load kwargs.json to get model configuration
            kwargs_path = Path(args.load_model) / 'kwargs.json'
            if not kwargs_path.is_file():
                raise FileNotFoundError(f"Cannot find kwargs.json in {args.load_model}")
            with open(kwargs_path, 'r') as f:
                loaded_kwargs = json.load(f)

            # 2. Create a temporary args object from loaded kwargs to pass to get_kwargs
            temp_args = argparse.Namespace(**loaded_kwargs)

            # 3. Reconstruct non-serializable objects (like vocab) using the loaded config
            if vocabulary_data is None:
                # In evaluation, vocabulary_data is not passed to construct_saver,
                # so we need to load it using the --training-data argument.
                if not hasattr(args, 'training_data') or args.training_data is None:
                    raise ValueError("--training-data is required to load vocabulary for evaluation.")
                vocabulary_data = load_vocabulary_data(args, self.parser)

            # 4. Call get_kwargs to reconstruct vocab, fsa_container, etc.
            full_kwargs = self.get_kwargs(temp_args, vocabulary_data)

            # 5. Construct model with full kwargs, then load weights into it
            saver = read_saver(self.construct_model, args.load_model, args.load_parameters, device, **full_kwargs)
            if self.use_output and args.output is not None:
                saver = saver.to_directory(args.output)
                saver.check_output()
        else:
            # --- TRAINING PATH ---
            if self.use_init:
                kwargs = self.get_kwargs(args, vocabulary_data)
                output = args.output
                # Construct the saver, which also constructs the model
                saver = construct_saver(self.construct_model, output, **kwargs)

                # After model construction, remove non-serializable objects before saving.
                for key in ['fsa', 'fsa_container', 'fsa_alphabet', 'word_vocab']:
                    if key in saver.kwargs:
                       del saver.kwargs[key]
                if 'reset_symbol_ids' in saver.kwargs and saver.kwargs['reset_symbol_ids'] is not None:
                    saver.kwargs['reset_symbol_ids'] = sorted(list(saver.kwargs['reset_symbol_ids']))

                saver.save_kwargs()
                saver.model.to(device)

                self.parameter_seed = args.parameter_seed
                if self.parameter_seed is None:
                    self.parameter_seed = random.getrandbits(32)
                if device.type == 'cuda':
                    torch.manual_seed(self.parameter_seed)
                    param_generator = None
                else:
                    param_generator = torch.manual_seed(self.parameter_seed)
                    self.initialize(args, saver.model, param_generator)
            else:
                raise ValueError("Either --load-model must be specified or the interface must be in 'init' mode.")

        self.on_saver_constructed(args, saver)
        return saver

    def construct_model(self, architecture=None, **kwargs):
        # When loading a saved model, `architecture` will be inside kwargs.
        if architecture is None:
            architecture = kwargs.get('architecture')

        if architecture is None:
            raise ValueError("Architecture is missing and could not be found in kwargs.")

        if architecture == 'hybrid_csg':
            return self._construct_hybrid_model(**kwargs)
        else:
            return self._construct_standard_model(architecture=architecture, **kwargs)

    def _construct_hybrid_model(self, hybrid_base_architecture, embedding_size, num_layers, dropout, d_model, num_heads, feedforward_size, hidden_units, input_vocabulary_size, lba_hidden_size, lba_n_steps, **kwargs):
        if hybrid_base_architecture is None: raise ValueError("For hybrid_csg, --hybrid-base-architecture must be specified.")
        if embedding_size is None: raise ValueError("For hybrid_csg, --embedding-size must be specified.")
        if dropout is None: raise ValueError("--dropout is required for hybrid_csg")
        if num_layers is None: raise ValueError("--num-layers is required for hybrid_csg")
        
        if hybrid_base_architecture == 'transformer':
            if d_model is None or num_heads is None or feedforward_size is None: raise ValueError("Transformer parameters are required for hybrid transformer.")
            if embedding_size != d_model: raise ValueError("For hybrid transformer, --embedding-size must equal --d-model.")
            
            main_model = UnidirectionalTransformerEncoderLayers(
                num_layers=num_layers, d_model=d_model, num_heads=num_heads,
                feedforward_size=feedforward_size, dropout=dropout, use_final_layer_norm=True
            )
            main_model_output_size = d_model
        
        elif hybrid_base_architecture in ('rnn', 'lstm'):
            if hidden_units is None: raise ValueError("--hidden-units is required for hybrid RNN/LSTM.")
            
            RecurrentModel = SimpleRNN if hybrid_base_architecture == 'rnn' else LSTM
            core = RecurrentModel(input_size=embedding_size, hidden_units=hidden_units, layers=num_layers, dropout=dropout, learned_hidden_state=True)
            main_model = core.main() @ DropoutUnidirectional(dropout)
            main_model_output_size = hidden_units
        else:
            raise ValueError(f"Unknown hybrid base architecture: {hybrid_base_architecture}")

        return HybridCSGModel(
            input_vocabulary_size=input_vocabulary_size, embedding_size=embedding_size,
            main_model=main_model, main_model_output_size=main_model_output_size,
            lba_hidden_size=lba_hidden_size, lba_n_steps=lba_n_steps,
            device=self.get_device(None)
        )

    def _construct_standard_model(self, architecture, add_ngram_head_n, num_layers, d_model, num_heads, feedforward_size, dropout, hidden_units, use_language_modeling_head, use_next_symbols_head, input_vocabulary_size, output_vocabulary_size, positional_encoding, reset_symbol_ids, use_fsa_features=False, fsa_container=None, fsa_alphabet=None, word_vocab=None, fsa_name=None, fsa_embedding_dim=None, **kwargs):
        if use_fsa_features and fsa_container is None:
            # Reconstruct FSA if loading from saved model
            fsa_func_name = f'{fsa_name}_structural_fsa_container'
            if not hasattr(structural_fsas, fsa_func_name):
                raise ValueError(f"Cannot reconstruct unknown FSA: {fsa_name}")
            fsa_func = getattr(structural_fsas, fsa_func_name)
            fsa_container, fsa_alphabet = fsa_func()

        if use_fsa_features and word_vocab is None:
            # This should not happen if get_kwargs is called correctly before this.
            # But as a safeguard when loading a model, we might need to reconstruct vocab.
            # For now, we rely on it being passed during training.
            raise ValueError("word_vocab is required when using FSA features, but it was not reconstructed.")

        core_pipeline = None
        output_size = 0
        shared_embeddings = None

        if architecture == 'transformer':
            if num_layers is None or d_model is None or num_heads is None or feedforward_size is None or dropout is None:
                raise ValueError("Transformer parameters are required.")
            
            shared_embeddings = get_shared_embeddings(
                tie_embeddings=use_language_modeling_head, input_vocabulary_size=input_vocabulary_size,
                output_vocabulary_size=output_vocabulary_size, embedding_size=d_model, use_padding=False
            )
            if use_fsa_features:
                full_input_layer = FSAIntegratedInputLayer(
                    word_vocab=word_vocab,
                    fsa_container=fsa_container,
                    fsa_alphabet=fsa_alphabet,
                    word_embedding_dim=d_model,
                    fsa_embedding_dim=fsa_embedding_dim,
                    output_dim=d_model,
                    use_padding=False,
                    dropout=dropout
                )
            else:
                if positional_encoding == 'resettable':
                    full_input_layer = ResettablePositionalInputLayer(
                        vocabulary_size=input_vocabulary_size, d_model=d_model, reset_symbols=reset_symbol_ids,
                        dropout=dropout, use_padding=False, shared_embeddings=shared_embeddings
                    )
                else:
                    full_input_layer = get_transformer_input_unidirectional(
                        vocabulary_size=input_vocabulary_size, d_model=d_model, dropout=dropout,
                        use_padding=False, shared_embeddings=shared_embeddings
                    )
            
            encoder = UnidirectionalTransformerEncoderLayers(
                num_layers=num_layers, d_model=d_model, num_heads=num_heads,
                feedforward_size=feedforward_size, dropout=dropout, use_final_layer_norm=True
            ).main()
            wrapped_encoder = ForceIncludeFirstFalse(encoder)

            core_pipeline = (
                full_input_layer @
                wrapped_encoder
            )
            output_size = d_model

        elif architecture in ('rnn', 'lstm'):
            if hidden_units is None or num_layers is None or dropout is None:
                raise ValueError("RNN/LSTM parameters are required.")

            shared_embeddings = get_shared_embeddings(
                tie_embeddings=use_language_modeling_head, input_vocabulary_size=input_vocabulary_size,
                output_vocabulary_size=output_vocabulary_size, embedding_size=hidden_units, use_padding=False
            )
            RecurrentModel = SimpleRNN if architecture == 'rnn' else LSTM
            core = RecurrentModel(input_size=hidden_units, hidden_units=hidden_units, layers=num_layers, dropout=dropout, learned_hidden_state=True)
            core_pipeline = (
                EmbeddingUnidirectional(
                    vocabulary_size=input_vocabulary_size, output_size=hidden_units,
                    use_padding=False, shared_embeddings=shared_embeddings
                ) @
                DropoutUnidirectional(dropout) @
                core.main() @
                DropoutUnidirectional(dropout)
            )
            output_size = hidden_units
        else:
            raise ValueError(f"Unknown standard architecture: {architecture}")
        
        if add_ngram_head_n > 0:
            ngram_head_layer = NgramHead(n=add_ngram_head_n, d_model=output_size)
            core_pipeline = core_pipeline @ Composable(ngram_head_layer).tag('ngram_head')

        return (
            Composable(core_pipeline).tag('core') @
            Composable(
                OutputHeads(
                    input_size=output_size, use_language_modeling_head=use_language_modeling_head,
                    use_next_symbols_head=use_next_symbols_head, vocabulary_size=output_vocabulary_size,
                    shared_embeddings=shared_embeddings
                )
            ).tag('output_heads')
        )

    def initialize(self, args, model, generator):
        if args.init_scale is None:
            raise ValueError
        smart_init(model, generator, fallback=uniform_fallback(args.init_scale))

    def on_saver_constructed(self, args, saver):
        # First, let the base class do its setup
        super().on_saver_constructed(args, saver)

        # Now, remove non-serializable arguments from the kwargs that will be saved.
        # This prevents the JSON serialization error during training,
        # while ensuring the objects were available during model construction.
        for key in ['fsa', 'fsa_container', 'fsa_alphabet', 'word_vocab']:
            if key in saver.kwargs:
                del saver.kwargs[key]

        if 'reset_symbol_ids' in saver.kwargs and saver.kwargs['reset_symbol_ids'] is not None:
            saver.kwargs['reset_symbol_ids'] = sorted(list(saver.kwargs['reset_symbol_ids']))

        self.bos_index = saver.kwargs['bos_index']
        self.uses_bos = self.bos_index is not None
        self.eos_index = saver.kwargs['eos_index']
        self.uses_eos = self.eos_index is not None
        self.use_language_modeling_head = saver.kwargs['use_language_modeling_head']
        self.use_next_symbols_head = saver.kwargs['use_next_symbols_head']
        self.add_ngram_head_n = saver.kwargs.get('add_ngram_head_n', 0)
        self.architecture = saver.kwargs.get('architecture')

        if self.use_language_modeling_head:
            self.output_padding_index = saver.kwargs['output_vocabulary_size']
        else:
            self.output_padding_index = None
        if self.use_next_symbols_head:
            self.output_vocabulary_size = saver.kwargs['output_vocabulary_size']
        else:
            self.output_vocabulary_size = None

    def adjust_length(self, length):
        return int(self.uses_bos) + length

    def get_vocabularies(self, vocabulary_data, builder=None):
        return get_vocabularies(vocabulary_data, self.uses_bos, self.uses_eos, builder)

    def prepare_batch(self, batch, device):
        if self.output_padding_index is not None:
            padding_index = self.output_padding_index
        else:
            padding_index = 0
            
        full_tensor, last_index = pad_sequences(
            [x[0] for x in batch], device,
            bos=self.bos_index, eos=self.eos_index,
            pad=padding_index, return_lengths=True
        )
        input_tensor = full_tensor[:, :-1] if self.eos_index is not None else full_tensor
        
        recognition_expected_tensor = torch.tensor([x[1][0] for x in batch], device=device, dtype=torch.float)
        
        positive_mask = None
        positive_output_lengths = None
        if self.use_language_modeling_head or self.use_next_symbols_head:
            positive_mask = recognition_expected_tensor.bool()
            positive_output_lengths = last_index[positive_mask] + 1

        language_modeling_expected_tensor = None
        if self.use_language_modeling_head:
            language_modeling_expected_tensor = full_tensor[:, 1:] if self.uses_bos else full_tensor
            language_modeling_expected_tensor = language_modeling_expected_tensor[positive_mask]

        next_symbols_expected_tensor = None
        next_symbols_padding_mask = None
        if self.use_next_symbols_head:
            next_symbols_data = [x[1][1] for x in batch if x[1][1] is not None]
            num_positive_examples = len(next_symbols_data)
            max_output_length = full_tensor.size(1) - int(self.uses_bos)
            next_symbols_expected_tensor = torch.zeros(
                (num_positive_examples, max_output_length, self.output_vocabulary_size), device=device
            )
            next_symbols_padding_mask = torch.zeros(
                (num_positive_examples, max_output_length), device=device
            )
            for i, next_symbol_set_list in enumerate(next_symbols_data):
                for j, next_symbol_set in enumerate(next_symbol_set_list):
                    next_symbols_expected_tensor[i, j, next_symbol_set] = 1
                next_symbols_padding_mask[i, :len(next_symbol_set_list)] = 1
        
        if not self.uses_bos and padding_index == self.output_padding_index:
            input_tensor = input_tensor.clone()
            input_tensor[input_tensor == self.output_padding_index] = 0
            
        return (
            ModelInput(input_tensor, last_index, positive_mask, input_tensor),
            (
                recognition_expected_tensor,
                language_modeling_expected_tensor,
                next_symbols_expected_tensor,
                next_symbols_padding_mask,
                positive_output_lengths
            )
        )

    def on_before_process_pairs(self, saver, datasets):
        arch = saver.kwargs.get('architecture')
        if arch == 'transformer' or (arch == 'hybrid_csg' and saver.kwargs.get('hybrid_base_architecture') == 'transformer'):
            max_length = max(len(x[0]) for dataset in datasets for x in dataset)
            self._preallocate_positional_encodings(saver, self.adjust_length(max_length))

    def _preallocate_positional_encodings(self, saver, max_length):
        d_model = saver.kwargs['d_model']
        for module in saver.model.modules():
            if isinstance(module, SinusoidalPositionalEncodingCacher):
                module.get_encodings(max_length, d_model)
                module.set_allow_reallocation(False)

    def get_logits(self, model, model_input):
        if self.architecture == 'hybrid_csg':
            return model(
                model_input.input_sequence,
                last_index=model_input.last_index,
                positive_mask=model_input.positive_mask
            )
            
        core_kwargs = {'include_first': not self.uses_bos}
        core_internal_tag_kwargs = {}
        if self.add_ngram_head_n > 0:
            core_internal_tag_kwargs['ngram_head'] = {'input_ids': model_input.input_ids}
        
        if core_internal_tag_kwargs:
            core_kwargs['tag_kwargs'] = core_internal_tag_kwargs
            
        tag_kwargs = dict(
            core=core_kwargs,
            output_heads=dict(
                last_index=model_input.last_index,
                positive_mask=model_input.positive_mask
            )
        )

        # For transformers, UnidirectionalTransformerEncoderLayers requires
        # include_first=False. The logic `not self.uses_bos` correctly sets this,
        # but the argument can get lost in the composition framework. We pass it
        # explicitly to the model call to ensure it is propagated correctly.
        if self.architecture == 'transformer':
            return model(model_input.input_sequence, include_first=False, tag_kwargs=tag_kwargs)
        else:
            return model(model_input.input_sequence, tag_kwargs=tag_kwargs)


class OutputHeads(torch.nn.Module):
    def __init__(self, input_size: int, use_language_modeling_head: bool, use_next_symbols_head: bool, vocabulary_size: int, shared_embeddings: torch.Tensor):
        super().__init__()
        self.recognition_head = Layer(input_size, 1, bias=True)
        self.language_modeling_head = None
        if use_language_modeling_head:
            self.language_modeling_head = OutputUnidirectional(
                input_size=input_size, vocabulary_size=vocabulary_size,
                shared_embeddings=shared_embeddings, bias=False
            )
        self.next_symbols_head = None
        if use_next_symbols_head:
            self.next_symbols_head = OutputUnidirectional(
                input_size=input_size, vocabulary_size=vocabulary_size, bias=True
            )

    def forward(self, inputs, last_index, positive_mask):
        last_inputs = torch.gather(
            inputs, 1,
            last_index[:, None, None].expand(-1, -1, inputs.size(2))
        ).squeeze(1)
        recognition_logit = self.recognition_head(last_inputs).squeeze(1)
        
        language_modeling_logits = None
        next_symbols_logits = None
        
        if self.language_modeling_head is not None or self.next_symbols_head is not None:
            positive_inputs = inputs[positive_mask]
        
        if self.language_modeling_head is not None:
            language_modeling_logits = self.language_modeling_head(positive_inputs, include_first=False)
        
        if self.next_symbols_head is not None:
            next_symbols_logits = self.next_symbols_head(positive_inputs, include_first=False)
            
        return recognition_logit, language_modeling_logits, next_symbols_logits
