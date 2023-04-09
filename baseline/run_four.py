# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

import logging
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import random
from tqdm import tqdm, trange
import csv
import glob
import json
#import apex

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import *
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
import xlrd

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)


class gaokaoExamples(object):
    def __init__(self, id, question, evis, options, label=None):
        self.id = id
        self.question = question
        self.evis = evis
        self.options = options
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            f"id: {self.id}",
            f"question: {self.question}",
            f"option_0: {self.options[0]}",
            f"option_1: {self.options[1]}",
            f"option_2: {self.options[2]}",
            f"option_3: {self.options[3]}",
            f"evi_0:{self.evis[0]}",
            f"evi_1:{self.evis[1]}",
            f"evi_2:{self.evis[2]}",
            f"evi_3:{self.evis[3]}",
        ]

        if self.label is not None:
            l.append(f"label: {self.label}")

        return ", ".join(l)


class dreamExamples(object):
    def __init__(self,
                 id,
                 context_sentence,
                 start_ending,
                 ending_0,
                 ending_1,
                 ending_2,
                 label=None):
        self.id = id
        self.context_sentence = context_sentence
        self.start_ending = start_ending
        self.endings = [
            ending_0,
            ending_1,
            ending_2,
        ]
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            f"id: {self.id}",
            f"article: {self.context_sentence}",
            f"question: {self.start_ending}",
            f"option_0: {self.endings[0]}",
            f"option_1: {self.endings[1]}",
            f"option_2: {self.endings[2]}",
        ]

        if self.label is not None:
            l.append(f"label: {self.label}")

        return ", ".join(l)


class generalExamples(object):
    def __init__(self,
                 q_id,
                 context_sentence,
                 start_ending,
                 op_a,
                 op_b,
                 op_c,
                 op_d,
                 label=None):

        self.id = q_id,
        self.context_sentence = context_sentence,
        self.endings = [op_a, op_b, op_c, op_d],
        self.question = start_ending,
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            f"id: {self.id}",
            f"article:{self.context_sentence}",
            f"question:{self.question}",
            f"choices:{self.endings}",
        ]
        if self.label is not None:
            l.append(f"label:{self.label}")
        return ",".join(l)


class InputFeatures(object):
    def __init__(self, example_id, choices_features, label):
        self.example_id = example_id
        self.choices_features = [{
            'input_ids': input_ids,
            'input_mask': input_mask,
            'segment_ids': segment_ids
        } for _, input_ids, input_mask, segment_ids in choices_features]
        self.label = label

def change_options(num):
    if num ==0:
        return 'A'
    if num ==1:
        return 'B'
    if num ==2:
        return 'C'
    if num ==3:
        return 'D'

def read_GCRC_train_examples(paths):
    examples = []
    with open(paths, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for id, line in enumerate(reader):
            if id == 0:
                continue
            else:
                q_id = line[0]
                passage = line[2]
                question = line[3]
                option_a, option_b, option_c, option_d = line[4][0], line[4][1], line[4][2], line[4][3]
                answer = line[5]
                examples.append(
                    generalExamples(q_id=q_id,
                                    context_sentence=passage,
                                    start_ending=question,
                                    op_a=option_a,
                                    op_b=option_b,
                                    op_c=option_c,
                                    op_d=option_d,
                                    label=answer))
    return examples

def read_robust_dev_examples(paths):

    examples_o = []
    examples_p = []
    examples_n = []
    with open(paths, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for id, line in enumerate(reader):
            if id == 0:
                continue
            else:
                q_id_o = line[0]
                passage_o = line[2]
                question_o = line[3]
                option_a_o, option_b_o, option_c_o, option_d_o = line[4][0], line[4][1], line[4][2], line[4][3]
                answer_o = line[5]
                examples_o.append(
                    generalExamples(q_id=q_id_o,
                                    context_sentence=passage_o,
                                    start_ending=question_o,
                                    op_a=option_a_o,
                                    op_b=option_b_o,
                                    op_c=option_c_o,
                                    op_d=option_d_o,
                                    label=answer_o))
                q_id_p = line[0]
                passage_p = line[2]
                question_p = line[3]
                option_a_p, option_b_p, option_c_p, option_d_p = line[6][0], line[6][1], line[6][2], line[6][3]
                answer_p = line[7]
                examples_p.append(
                    generalExamples(q_id=q_id_p,
                                    context_sentence=passage_p,
                                    start_ending=question_p,
                                    op_a=option_a_p,
                                    op_b=option_b_p,
                                    op_c=option_c_p,
                                    op_d=option_d_p,
                                    label=answer_p))
                q_id_n = line[0]
                passage_n = line[2]
                question_n = line[8]
                option_a_n, option_b_n, option_c_n, option_d_n = line[9][0], line[9][1], line[9][2], line[9][3]
                answer_n = line[10]
                examples_n.append(
                    generalExamples(q_id=q_id_n,
                                    context_sentence=passage_n,
                                    start_ending=question_n,
                                    op_a=option_a_n,
                                    op_b=option_b_n,
                                    op_c=option_c_n,
                                    op_d=option_d_n,
                                    label=answer_n))
    return examples_o,examples_p,examples_n

def read_robust_test_examples(paths):

    examples_o = []
    examples_p = []
    examples_n = []
    with open(paths, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for id, line in enumerate(reader):
            if id == 0:
                continue
            else:
                q_id_o = line[0]
                passage_o = line[2]
                question_o = line[3]
                option_a_o, option_b_o, option_c_o, option_d_o = line[4][0], line[4][1], line[4][2], line[4][3]
                answer_o = ''
                examples_o.append(
                    generalExamples(q_id=q_id_o,
                                    context_sentence=passage_o,
                                    start_ending=question_o,
                                    op_a=option_a_o,
                                    op_b=option_b_o,
                                    op_c=option_c_o,
                                    op_d=option_d_o,
                                    label=answer_o))
                q_id_p = line[0]
                passage_p = line[2]
                question_p = line[3]
                option_a_p, option_b_p, option_c_p, option_d_p = line[5][0], line[5][1], line[5][2], line[5][3]
                answer_p = ''
                examples_p.append(
                    generalExamples(q_id=q_id_p,
                                    context_sentence=passage_p,
                                    start_ending=question_p,
                                    op_a=option_a_p,
                                    op_b=option_b_p,
                                    op_c=option_c_p,
                                    op_d=option_d_p,
                                    label=answer_p))
                q_id_n = line[0]
                passage_n = line[2]
                question_n = line[6]
                option_a_n, option_b_n, option_c_n, option_d_n = line[7][0], line[7][1], line[7][2], line[7][3]
                answer_n = ''
                examples_n.append(
                    generalExamples(q_id=q_id_n,
                                    context_sentence=passage_n,
                                    start_ending=question_n,
                                    op_a=option_a_n,
                                    op_b=option_b_n,
                                    op_c=option_c_n,
                                    op_d=option_d_n,
                                    label=answer_n))
    return examples_o,examples_p,examples_n

def read_general_examples(paths):
    examples = []
    with open(paths, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for id, line in enumerate(reader):
            if id == 0:
                continue
            else:
                q_id = line[1]
                passage = line[3]
                question = line[4]
                option_a, option_b, option_c, option_d = line[5], line[6], line[7], line[8]
                answer = line[9]
                examples.append(
                    generalExamples(q_id=q_id,
                                    context_sentence=passage,
                                    start_ending=question,
                                    op_a=option_a,
                                    op_b=option_b,
                                    op_c=option_c,
                                    op_d=option_d,
                                    label=answer))
    return examples

def read_dream_examples(path):
    examples = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for id, data in enumerate(reader):
            if id == 0:
                continue
            else:
                article = data[3]
                q_id = data[1]
                question = data[4]
                opA = data[5]
                opB = data[6]
                opC = data[7]
                answer = data[9]
                examples.append(
                    dreamExamples(id=q_id,
                                  context_sentence=article,
                                  start_ending=question,
                                  ending_0=opA,
                                  ending_1=opB,
                                  ending_2=opC,
                                  label=answer))

    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 is_training, task):
    """Loads a data file into a list of `InputBatch`s."""

    # RACE is a multiple choice task. To perform this task using Bert,
    # we will use the formatting proposed in "Improving Language
    # Understanding by Generative Pre-Training" and suggested by
    # @jacobdevlin-google in this issue
    # https://github.com/google-research/bert/issues/38.
    #
    # The input will be like:
    # [CLS] Article [SEP] Question + Option [SEP]
    # for each option
    #
    # The model will output a single value for each input. To get the
    # final decision of the model, we will run a softmax over these 4
    # outputs.

    label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

    features = []
    for example_index, example in enumerate(examples):
        # print(example_index)
        context_tokens = tokenizer.tokenize(''.join(example.context_sentence))
        question = tokenizer.tokenize(''.join(example.question))

        choices_features = []
        label = [label_map[i] for i in example.label]
        for index, option in enumerate(example.endings[0]):
            # We create a copy of the context tokens in order to be
            # able to shrink it according to ending_tokens
            # context = example.evis[index]


            ending_tokens = question + tokenizer.tokenize(''.join(option))

            context_tokens = context_tokens[:]
            _truncate_seq_pair(context_tokens, ending_tokens,
                               max_seq_length - 3)

            tokens = ["[CLS]"] + context_tokens + ["[SEP]"] + ending_tokens + [
                "[SEP]"
            ]
            segment_ids = [0] * (len(context_tokens) +
                                 2) + [1] * (len(ending_tokens) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            choices_features.append(
                (tokens, input_ids, input_mask, segment_ids))

        label = label_map[example.label]

        ## display some example
        if example_index < 1:
            logger.info("*** Example ***")
            logger.info(f"id: {example.id}")
            for choice_idx, (tokens, input_ids, input_mask,
                             segment_ids) in enumerate(choices_features):
                logger.info(f"choice: {choice_idx}")
                logger.info(f"tokens: {' '.join(tokens)}")
                logger.info(f"input_ids: {' '.join(map(str, input_ids))}")
                logger.info(f"input_mask: {' '.join(map(str, input_mask))}")
                logger.info(f"segment_ids: {' '.join(map(str, segment_ids))}")
            if is_training:
                logger.info(f"label: {label}")

        features.append(
            InputFeatures(example_id=example.id,
                          choices_features=choices_features,
                          label=label))

    return features

def convert_examples_to_features_test(examples, tokenizer, max_seq_length,
                                 is_training, task):
    """Loads a data file into a list of `InputBatch`s."""

    # RACE is a multiple choice task. To perform this task using Bert,
    # we will use the formatting proposed in "Improving Language
    # Understanding by Generative Pre-Training" and suggested by
    # @jacobdevlin-google in this issue
    # https://github.com/google-research/bert/issues/38.
    #
    # The input will be like:
    # [CLS] Article [SEP] Question + Option [SEP]
    # for each option
    #
    # The model will output a single value for each input. To get the
    # final decision of the model, we will run a softmax over these 4
    # outputs.

    label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

    features = []
    for example_index, example in enumerate(examples):
        # print(example_index)
        context_tokens = tokenizer.tokenize(''.join(example.context_sentence))
        question = tokenizer.tokenize(''.join(example.question))

        choices_features = []
        label = [label_map[i] for i in example.label]
        for index, option in enumerate(example.endings[0]):
            # We create a copy of the context tokens in order to be
            # able to shrink it according to ending_tokens
            # context = example.evis[index]


            ending_tokens = question + tokenizer.tokenize(''.join(option))

            context_tokens = context_tokens[:]
            _truncate_seq_pair(context_tokens, ending_tokens,
                               max_seq_length - 3)

            tokens = ["[CLS]"] + context_tokens + ["[SEP]"] + ending_tokens + [
                "[SEP]"
            ]
            segment_ids = [0] * (len(context_tokens) +
                                 2) + [1] * (len(ending_tokens) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            choices_features.append(
                (tokens, input_ids, input_mask, segment_ids))

        #label = label_map[example.label]

        ## display some example
        if example_index < 1:
            logger.info("*** Example ***")
            logger.info(f"id: {example.id}")
            for choice_idx, (tokens, input_ids, input_mask,
                             segment_ids) in enumerate(choices_features):
                logger.info(f"choice: {choice_idx}")
                logger.info(f"tokens: {' '.join(tokens)}")
                logger.info(f"input_ids: {' '.join(map(str, input_ids))}")
                logger.info(f"input_mask: {' '.join(map(str, input_mask))}")
                logger.info(f"segment_ids: {' '.join(map(str, segment_ids))}")
            if is_training:
                logger.info(f"label: {label}")

        features.append(
            InputFeatures(example_id=example.id,
                          choices_features=choices_features,
                          label=label))

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def accuracy(predict, labels):

    return np.sum(predict == labels), predict


def select_field(features, field):
    return [[choice[field] for choice in feature.choices_features]
            for feature in features]


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help=
        "The input data dir. Should contain the .csv files (or other data files) for the task."
    )
    parser.add_argument(
        "--bert_model",
        default=None,
        type=str,
        required=True,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese."
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints will be written."
    )

    ## Other parameters
    parser.add_argument(
        "--max_seq_length",
        default=450,
        type=int,
        help=
        "The maximum total input sequence length after WordPiece tokenization. \n"
        "Sequences longer than this will be truncated, and sequences shorter \n"
        "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        # required=True,
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        # required=True,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict",
                        default=False,
                        # required=True,
                        help="Whether to run eval on the test set.")
    parser.add_argument(
        "--do_lower_case",
        default=False,
        action='store_true',
        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=4,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--test_batch_size",
                        default=4,
                        type=int,
                        help="Total batch size for test.")
    parser.add_argument("--learning_rate",
                        default=1e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help=
        "Proportion of training to perform linear learning rate warmup for. "
        "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=4,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass."
    )
    parser.add_argument(
        '--fp16',
        default=False,
        action='store_true',
        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: ")
    parser.add_argument(
        '--loss_scale',
        type=float,
        default=0,
        help=
        "Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
        "0 (default value): dynamic loss scaling.\n"
        "Positive power of 2: static loss scaling value.\n")

    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available()
                              and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info(
        "device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".
        format(device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1"
            .format(args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size /
                                args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not os.path.exists(args.output_dir):
        print("Output directory ({}) is empty.".format(
                args.output_dir))
        os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model,
                                              do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_steps = None
    if args.do_train:
        train_dir = os.path.join(args.data_dir, 'train_GCRC.csv')
        if args.task_name != 'dream':

            train_examples = read_GCRC_train_examples(train_dir)

        else:
            train_examples = read_dream_examples(train_dir)

        num_train_steps = int(
            len(train_examples) / args.train_batch_size /
            args.gradient_accumulation_steps * args.num_train_epochs)

    if (args.task_name) != 'dream':
        num_choices = 4
    else:
        num_choices = 3

    # Prepare model
    model = BertForMultipleChoice.from_pretrained(
        args.bert_model,
        cache_dir=PYTORCH_PRETRAINED_BERT_CACHE /
        'distributed_{}'.format(args.local_rank),
        num_choices=num_choices)
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{
        'params':
        [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay':
        0.01
    }, {
        'params':
        [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay':
        0.0
    }]
    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer,
                                       static_loss_scale=args.loss_scale)
    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)
    task = args.task_name
    global_step = 0
    if args.do_train:
        train_features = convert_examples_to_features(train_examples,
                                                      tokenizer,
                                                      args.max_seq_length,
                                                      True, task)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        all_input_ids = torch.tensor(select_field(train_features, 'input_ids'),dtype=torch.long)
        all_input_mask = torch.tensor(select_field(train_features,'input_mask'),dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(train_features,'segment_ids'),dtype=torch.long)
        all_label = torch.tensor([f.label for f in train_features],dtype=torch.long)
        # print(all_label.size(0))
        # print(all_segment_ids.size(0))
        train_data = TensorDataset(all_input_ids, all_input_mask,
                                   all_segment_ids, all_label)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data,
                                      sampler=train_sampler,
                                      batch_size=args.train_batch_size)
        best_acc = 0.0
        model.train()
        for ep in range(int(args.num_train_epochs)):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            logger.info("Trianing Epoch: {}/{}".format(
                ep + 1, int(args.num_train_epochs)))
            for step, batch in enumerate(tqdm(train_dataloader)):
                # step:id of batch
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                loss = model(input_ids, segment_ids, input_mask, label_ids)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.fp16 and args.loss_scale != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * args.loss_scale
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * warmup_linear(
                        global_step / t_total, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                if global_step % 10 == 0:
                    # if global_step % 100 == 0:
                    logger.info("Training loss: {}, global step: {}".format(
                        tr_loss / nb_tr_steps, global_step))

            if args.do_eval:
                dev_dir = os.path.join(args.data_dir, 'dev_GCRCadvrobust.csv')

                if task != 'dream':

                    eval_examples_o, eval_examples_p, eval_examples_n = read_robust_dev_examples(dev_dir)

                else:

                    eval_examples = read_dream_examples(dev_dir)

                # 保存
                # 从这开始分成三份就行
                # 所有id集合
                all_id = [f.id[0] for f in eval_examples_o]
                # 原始答案集合
                new_label_o = []
                new_predict_o = []
                # 正对抗答案集合
                new_label_p = []
                new_predict_p = []
                # 负对抗答案集合
                new_label_n = []
                new_predict_n = []
                eval_features_o = convert_examples_to_features(
                    eval_examples_o, tokenizer, args.max_seq_length, True, task)
                eval_features_p = convert_examples_to_features(
                    eval_examples_p, tokenizer, args.max_seq_length, True, task)
                eval_features_n = convert_examples_to_features(
                    eval_examples_n, tokenizer, args.max_seq_length, True, task)
                logger.info("***** Running evaluation: Dev *****")
                logger.info("  Num examples = %d", len(eval_examples_o))
                logger.info("  Batch size = %d", args.eval_batch_size)
                # 原始
                all_input_ids_o = torch.tensor(select_field(
                    eval_features_o, 'input_ids'),
                    dtype=torch.long)
                all_input_mask_o = torch.tensor(select_field(
                    eval_features_o, 'input_mask'),
                    dtype=torch.long)
                all_segment_ids_o = torch.tensor(select_field(
                    eval_features_o, 'segment_ids'),
                    dtype=torch.long)
                all_label_o = torch.tensor([f.label for f in eval_features_o],
                                           dtype=torch.long)

                eval_data_o = TensorDataset(all_input_ids_o, all_input_mask_o,
                                            all_segment_ids_o, all_label_o)
                # 正对抗
                all_input_ids_p = torch.tensor(select_field(
                    eval_features_p, 'input_ids'),
                    dtype=torch.long)
                all_input_mask_p = torch.tensor(select_field(
                    eval_features_p, 'input_mask'),
                    dtype=torch.long)
                all_segment_ids_p = torch.tensor(select_field(
                    eval_features_p, 'segment_ids'),
                    dtype=torch.long)
                all_label_p = torch.tensor([f.label for f in eval_features_p],
                                           dtype=torch.long)
                eval_data_p = TensorDataset(all_input_ids_p, all_input_mask_p,
                                            all_segment_ids_p, all_label_p)
                # 负对抗
                all_input_ids_n = torch.tensor(select_field(
                    eval_features_n, 'input_ids'),
                    dtype=torch.long)
                all_input_mask_n = torch.tensor(select_field(
                    eval_features_n, 'input_mask'),
                    dtype=torch.long)
                all_segment_ids_n = torch.tensor(select_field(
                    eval_features_n, 'segment_ids'),
                    dtype=torch.long)
                all_label_n = torch.tensor([f.label for f in eval_features_n],
                                           dtype=torch.long)
                eval_data_n = TensorDataset(all_input_ids_n, all_input_mask_n,
                                            all_segment_ids_n, all_label_n)
                # Run prediction for full data 原始
                eval_sampler_o = SequentialSampler(eval_data_o)
                eval_dataloader_o = DataLoader(eval_data_o,
                                               sampler=eval_sampler_o,
                                               batch_size=args.eval_batch_size)
                # Run prediction for full data 正对抗
                eval_sampler_p = SequentialSampler(eval_data_p)
                eval_dataloader_p = DataLoader(eval_data_p,
                                               sampler=eval_sampler_p,
                                               batch_size=args.eval_batch_size)
                # Run prediction for full data 负对抗
                eval_sampler_n = SequentialSampler(eval_data_n)
                eval_dataloader_n = DataLoader(eval_data_n,
                                               sampler=eval_sampler_n,
                                               batch_size=args.eval_batch_size)

                model.eval()
                eval_loss_o = 0
                eval_loss_p = 0
                eval_loss_n = 0
                # eval batch 原始
                for step, batch in enumerate(tqdm(eval_dataloader_o)):
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, segment_ids, label_ids = batch

                    with torch.no_grad():
                        tmp_eval_loss = model(input_ids, segment_ids,
                                              input_mask, label_ids)
                        logits = model(input_ids, segment_ids,
                                       input_mask)  # [batch_size,4]

                        logits = logits.detach().cpu().numpy()
                        label_ids = label_ids.to('cpu').numpy()
                        predict_labels = np.argmax(logits, axis=1)
                        for i in label_ids:
                            new_label_o.append(i)
                        for j in predict_labels:
                            new_predict_o.append(j)
                        eval_loss_o += tmp_eval_loss.mean().item()  # gpu平均loss

                # eval batch 正对抗
                for step, batch in enumerate(tqdm(eval_dataloader_p)):
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, segment_ids, label_ids = batch

                    with torch.no_grad():
                        tmp_eval_loss = model(input_ids, segment_ids,
                                              input_mask, label_ids)
                        logits = model(input_ids, segment_ids,
                                       input_mask)  # [batch_size,4]

                        logits = logits.detach().cpu().numpy()
                        label_ids = label_ids.to('cpu').numpy()
                        predict_labels = np.argmax(logits, axis=1)
                        for i in label_ids:
                            new_label_p.append(i)
                        for j in predict_labels:
                            new_predict_p.append(j)
                        tmp_eval_accuracy = accuracy(
                            predict_labels, label_ids)[0]  # 与真实label相等的
                        eval_loss_p += tmp_eval_loss.mean().item()  # gpu平均loss

                # eval batch 负对抗
                for step, batch in enumerate(tqdm(eval_dataloader_n)):
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, segment_ids, label_ids = batch

                    with torch.no_grad():
                        tmp_eval_loss = model(input_ids, segment_ids,
                                              input_mask, label_ids)
                        logits = model(input_ids, segment_ids,
                                       input_mask)  # [batch_size,4]

                        logits = logits.detach().cpu().numpy()
                        label_ids = label_ids.to('cpu').numpy()
                        predict_labels = np.argmax(logits, axis=1)
                        for i in label_ids:
                            new_label_n.append(i)
                        for j in predict_labels:
                            new_predict_n.append(j)
                        tmp_eval_accuracy = accuracy(
                            predict_labels, label_ids)[0]  # 与真实label相等的
                        eval_loss_n += tmp_eval_loss.mean().item()  # gpu平均loss

                # 组合ID和答案
                pred_o, gold_o = [], []
                pred_p, gold_p = [], []
                pred_n, gold_n = [], []

                for ide, id in enumerate(all_id):
                    gold_o.append((all_id[ide], new_label_o[ide]))
                    pred_o.append((all_id[ide], new_predict_o[ide]))
                    gold_p.append((all_id[ide], new_label_p[ide]))
                    pred_p.append((all_id[ide], new_predict_p[ide]))
                    gold_n.append((all_id[ide], new_label_n[ide]))
                    pred_n.append((all_id[ide], new_predict_n[ide]))
                # 结果保存

                # 计算roBust准确率
                # 原始选项
                pred_res_set_o, gold_res_set_o = set(pred_o), set(gold_o)
                # 正对抗选项
                pred_res_set_p, gold_res_set_p = set(pred_p), set(gold_p)
                # 负对抗选项
                pred_res_set_n, gold_res_set_n = set(pred_n), set(gold_n)
                # 原始选项正确集合
                correct_ori = pred_res_set_o & gold_res_set_o
                # 正对抗选项正确集合
                correct_posi = pred_res_set_p & gold_res_set_p
                # 负对抗选项正确集合
                correct_nega = pred_res_set_n & gold_res_set_n

                # 去答案编号
                correct_ori_list = []
                for i in correct_ori:
                    correct_ori_list.append(i[0])
                correct_ori_set = set(correct_ori_list)

                correct_posi_list = []
                for i in correct_posi:
                    correct_posi_list.append(i[0])
                correct_posi_set = set(correct_posi_list)

                correct_nega_list = []
                for i in correct_nega:
                    correct_nega_list.append(i[0])
                correct_nega_set = set(correct_nega_list)

                # 原始选项与正对抗选项正确交集
                mix_oripo = correct_ori_set & correct_posi_set

                # 原始选项与负对抗选项正确交集
                mix_orine = correct_ori_set & correct_nega_set

                #  原始始选项和对抗选项其中之一正确预测选项
                correct_adv1 = mix_oripo | mix_orine

                # 原始选项和两个对抗选项均正确预测选项
                correct_adv2 = mix_oripo & mix_orine

                ori_acc = 1.0 * len(correct_ori) / len(pred_res_set_o)
                posi_acc = 1.0 * len(correct_posi) / len(pred_res_set_o)
                nega_acc = 1.0 * len(correct_nega) / len(pred_res_set_o)
                adv_acc1 = 1.0 * len(correct_adv1) / len(pred_res_set_o)
                adv_acc2 = 1.0 * len(correct_adv2) / len(pred_res_set_o)
                Score = 0.2 * ori_acc + 0.3 * adv_acc1 + 0.5 * adv_acc2

                result = {
                    'Acc_0': ori_acc,
                    'posi_acc':posi_acc,
                    'nega_acc':nega_acc,
                    'Acc_1': adv_acc1,
                    'Acc_2': adv_acc2,
                    'Score': Score,
                    'dev_eval_loss_o': eval_loss_o,
                    'dev_eval_loss_p': eval_loss_p,
                    'dev_eval_loss_n': eval_loss_n,
                    'global_step': global_step,
                    'loss': tr_loss / nb_tr_steps
                }
                output_eval_file = os.path.join(args.output_dir,
                                                "eval_results.txt")
                with open(output_eval_file, "a+") as writer:
                    logger.info("***** Dev results *****")
                    for key in sorted(result.keys()):
                        logger.info("  %s = %s", key, str(result[key]))
                        writer.write("%s = %s\n" % (key, str(result[key])))
                if Score > best_acc:
                    best_acc = Score
                    model_to_save = model.module if hasattr(
                        model, 'module') else model
                    output_model_file = os.path.join(args.output_dir,
                                                     'pytorch_model.bin')
                    torch.save(model_to_save.state_dict(), output_model_file)
        # test
    #model.load_state_dict((torch.load(os.path.join(args.output_dir, "pytorch_model.bin"))))
    if args.do_predict:
        model.load_state_dict((torch.load(os.path.join(args.output_dir, "pytorch_model.bin"))))
        test_dir = os.path.join(args.data_dir, 'test_GCRCadvrobust_public.csv')
        if task != 'dream':
            test_examples_o, test_examples_p, test_examples_n = read_robust_test_examples(test_dir)
        else:
            test_examples = read_dream_examples(test_dir)
        test_features_o = convert_examples_to_features_test(
            test_examples_o, tokenizer, args.max_seq_length, True, task)
        test_features_p = convert_examples_to_features_test(
            test_examples_p, tokenizer, args.max_seq_length, True, task)
        test_features_n = convert_examples_to_features_test(
            test_examples_n, tokenizer, args.max_seq_length, True, task)
        logger.info("***** Running : test step *****")
        logger.info("  Num examples = %d", len(test_examples_o))
        logger.info("  Batch size = %d", args.test_batch_size)
        # 保存
        # 从这开始分成三份就行
        # 所有id集合
        all_id = [f.id[0] for f in test_examples_o]
        renwu = {"data": []}
        test_json_dir = os.path.join(args.data_dir, 'test_GCRCadvrobust_public.json')
        with open(test_json_dir, encoding="utf-8") as a:
            data = json.load(a)
            for k in data["data"]:
                t = {}
                t["id"] = k["id"]
                t["title"] = k["title"]
                t["passage"] = k["passage"]
                t["question"] = k["question"]
                t["options"] = k["options"]
                t["answer"] = ''
                t["positive_options"] = k["positive_options"]
                t["positive_answer"] = ''
                t["negative_question"] = k["negative_question"]
                t["negative_options"] = k["negative_options"]
                t["negative_answer"] = ''
                renwu["data"].append(t)
        # 原始答案集合
        new_label_o = []
        new_predict_o = []
        # 正对抗答案集合
        new_label_p = []
        new_predict_p = []
        # 负对抗答案集合
        new_label_n = []
        new_predict_n = []
        # 原始
        all_input_ids_o = torch.tensor(select_field(
            test_features_o, 'input_ids'),
            dtype=torch.long)
        all_input_mask_o = torch.tensor(select_field(
            test_features_o, 'input_mask'),
            dtype=torch.long)
        all_segment_ids_o = torch.tensor(select_field(
            test_features_o, 'segment_ids'),
            dtype=torch.long)
        all_label_o = torch.tensor([f.label for f in test_features_o],
                                   dtype=torch.long)

        test_data_o = TensorDataset(all_input_ids_o, all_input_mask_o,
                                    all_segment_ids_o, all_label_o)
        # 正对抗
        all_input_ids_p = torch.tensor(select_field(
            test_features_p, 'input_ids'),
            dtype=torch.long)
        all_input_mask_p = torch.tensor(select_field(
            test_features_p, 'input_mask'),
            dtype=torch.long)
        all_segment_ids_p = torch.tensor(select_field(
            test_features_p, 'segment_ids'),
            dtype=torch.long)
        all_label_p = torch.tensor([f.label for f in test_features_p],
                                   dtype=torch.long)
        test_data_p = TensorDataset(all_input_ids_p, all_input_mask_p,
                                    all_segment_ids_p, all_label_p)
        # 负对抗
        all_input_ids_n = torch.tensor(select_field(
            test_features_n, 'input_ids'),
            dtype=torch.long)
        all_input_mask_n = torch.tensor(select_field(
            test_features_n, 'input_mask'),
            dtype=torch.long)
        all_segment_ids_n = torch.tensor(select_field(
            test_features_n, 'segment_ids'),
            dtype=torch.long)
        all_label_n = torch.tensor([f.label for f in test_features_n],
                                   dtype=torch.long)
        test_data_n = TensorDataset(all_input_ids_n, all_input_mask_n,
                                    all_segment_ids_n, all_label_n)
        # Run prediction for full data 原始
        test_sampler_o = SequentialSampler(test_data_o)
        test_dataloader_o = DataLoader(test_data_o,
                                       sampler=test_sampler_o,
                                       batch_size=args.test_batch_size)
        # Run prediction for full data 正对抗
        test_sampler_p = SequentialSampler(test_data_p)
        test_dataloader_p = DataLoader(test_data_p,
                                       sampler=test_sampler_p,
                                       batch_size=args.test_batch_size)
        # Run prediction for full data 负对抗
        test_sampler_n = SequentialSampler(test_data_n)
        test_dataloader_n = DataLoader(test_data_n,
                                       sampler=test_sampler_n,
                                       batch_size=args.test_batch_size)

        model.eval()

        # test batch 原始
        for step, batch in enumerate(tqdm(test_dataloader_o)):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            with torch.no_grad():
                logits = model(input_ids, segment_ids,
                               input_mask)  # [batch_size,4]

                logits = logits.detach().cpu().numpy()
                predict_labels = np.argmax(logits, axis=1)
                for j in predict_labels:
                    new_predict_o.append(j)

        # test batch 正对抗
        for step, batch in enumerate(tqdm(test_dataloader_p)):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            with torch.no_grad():
                logits = model(input_ids, segment_ids,
                               input_mask)  # [batch_size,4]

                logits = logits.detach().cpu().numpy()
                predict_labels = np.argmax(logits, axis=1)
                for j in predict_labels:
                    new_predict_p.append(j)

        # test batch 负对抗
        for step, batch in enumerate(tqdm(test_dataloader_n)):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            with torch.no_grad():
                logits = model(input_ids, segment_ids,
                               input_mask)  # [batch_size,4]

                logits = logits.detach().cpu().numpy()
                predict_labels = np.argmax(logits, axis=1)
                for j in predict_labels:
                    new_predict_n.append(j)

        for ide, id in enumerate(all_id):
            renwu["data"][ide]["answer"] = change_options(new_predict_o[ide])
            renwu["data"][ide]["positive_answer"] = change_options(new_predict_p[ide])
            renwu["data"][ide]["negative_answer"] = change_options(new_predict_n[ide])
        # 结果保存
        output_test_file = os.path.join(args.output_dir,'GCRC_advRobust.json')
        print("结果保存")
        with open(output_test_file, 'w', encoding="utf-8") as c:
            json.dump(renwu, c, ensure_ascii=False, indent=4)




if __name__ == "__main__":
    main()