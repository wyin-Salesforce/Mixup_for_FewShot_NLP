# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
import codecs
import json
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from scipy.stats import beta
from torch.nn import CrossEntropyLoss, MSELoss
from scipy.special import softmax
from mixup import tile
from transformers.tokenization_roberta import RobertaTokenizer
from transformers.optimization import AdamW
from transformers.modeling_roberta import RobertaModel#RobertaForSequenceClassification


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

# from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertModel
# import torch.nn as nn

bert_hidden_dim = 1024
pretrain_model_dir = 'roberta-large' #'roberta-large' , 'roberta-large-mnli', 'bert-large-uncased'



def store_transformers_models(model, tokenizer, output_dir, flag_str):
    '''
    store the model
    '''
    output_dir+='/'+flag_str
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    print('starting model storing....')
    # model.save_pretrained(output_dir)
    torch.save(model.state_dict(), output_dir)
    # tokenizer.save_pretrained(output_dir)
    print('store succeed')

class RobertaForSequenceClassification(nn.Module):
    def __init__(self, tagset_size):
        super(RobertaForSequenceClassification, self).__init__()
        self.tagset_size = tagset_size

        self.roberta_single= RobertaModel.from_pretrained(pretrain_model_dir)
        self.hidden_layer_0 = nn.Linear(bert_hidden_dim*3, bert_hidden_dim)
        self.hidden_layer_1 = nn.Linear(bert_hidden_dim, bert_hidden_dim)
        self.hidden_layer_2 = nn.Linear(bert_hidden_dim, bert_hidden_dim)
        self.single_hidden2tag = RobertaClassificationHead(bert_hidden_dim, tagset_size)

    def forward(self, input_ids, input_mask, span_a_mask, span_b_mask, lambda_value, is_train = False):
        # single_train_input_ids, single_train_input_mask, single_train_segment_ids, single_train_label_ids = batch_single
        outputs_single = self.roberta_single(input_ids, input_mask, None)
        output_last_layer_tensor3 = outputs_single[0] #(batch_size, sequence_length, hidden_size)`)
        span_a_reps = torch.sum(output_last_layer_tensor3*span_a_mask.unsqueeze(2), dim=1) #(batch, hidden)
        span_b_reps = torch.sum(output_last_layer_tensor3*span_b_mask.unsqueeze(2), dim=1) #(batch, hidden)
        combined_rep = torch.cat([span_a_reps, span_b_reps, span_a_reps*span_b_reps],dim=1) #(batch, 3*hidden)
        MLP_input = torch.tanh(self.hidden_layer_0(combined_rep))#(batch, hidden)

        hidden_states_single = torch.tanh(self.hidden_layer_2(torch.tanh(self.hidden_layer_1(MLP_input)))) #(batch, hidden)

        if is_train:
            batch_size = input_ids.shape[0]#.cpu().numpy()
            mixed_reps_matrix = torch.mm(lambda_value, hidden_states_single) #(mix_times, hidden_dim)
            score_single = self.single_hidden2tag(mixed_reps_matrix) #(batch, tag_set)
            return score_single
        else:
            score_single = self.single_hidden2tag(hidden_states_single) #(batch, tag_set)
            return score_single



class RobertaClassificationHead(nn.Module):
    """wenpeng overwrite it so to accept matrix as input"""

    def __init__(self, bert_hidden_dim, num_labels):
        super(RobertaClassificationHead, self).__init__()
        # self.dense = nn.Linear(bert_hidden_dim, bert_hidden_dim)
        # self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(bert_hidden_dim, num_labels)

    def forward(self, features):
        x = features#[:, 0, :]  # take <s> token (equiv. to [CLS])
        # x = self.dropout(x)
        # x = self.dense(x)
        # x = torch.tanh(x)
        # x = self.dropout(x)
        x = self.out_proj(x)
        return x

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a=None, span_a_left=None, span_a_right=None, text_b=None, span_b_left=None, span_b_right=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.span_a_left = span_a_left
        self.span_a_right = span_a_right

        self.text_b = text_b
        self.span_b_left = span_b_left
        self.span_b_right = span_b_right
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, span_a_mask, span_b_mask, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.span_a_mask = span_a_mask
        self.span_b_mask = span_b_mask
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def load_FewRel_data(self, k_shot):
        dev_relation_2_examples = {}
        filenames = ['train_wiki.json', 'val_wiki.json']
        for filename in filenames:
            with open('/export/home/Dataset/FewRel.1.0/'+filename) as json_file:
                dev_data = json.load(json_file)
                for relation, example_list in dev_data.items():
                    assert len(example_list) == 700
                    tup_list = []
                    for example in example_list:
                        sent = ' '.join(example.get('tokens'))
                        head_left = example.get('h')[2][0][0]
                        head_right = example.get('h')[2][0][-1]
                        tail_left = example.get('t')[2][0][0]
                        tail_right = example.get('t')[2][0][-1]
                        tup_list.append((sent, head_left, head_right, tail_left, tail_right))
                    assert len(tup_list) == 700
                    dev_relation_2_examples[relation] = tup_list
            json_file.close()

        relation_list = list(dev_relation_2_examples.keys())
        assert len(relation_list) == 80
        dev_4_train = {}
        dev_4_dev = {}
        dev_4_test = {}
        '''400, 100, 200'''
        for relation, ex_list in dev_relation_2_examples.items():
            random.shuffle(ex_list)
            dev_4_train[relation] = ex_list[:400]
            dev_4_dev[relation] = ex_list[400:500]
            dev_4_test[relation] = ex_list[500:]

        selected_dev_4_train = {}
        for relation, ex_list in dev_4_train.items():
            select_size = int(len(ex_list)*k_shot)
            selected_dev_4_train[relation] = ex_list if k_shot == 0 else random.sample(ex_list, select_size)

        '''build train'''
        train_examples = []
        ex_id = 0
        for relation, example_list in selected_dev_4_train.items():
            for example in example_list:
                sentence, head_left, head_right, tail_left, tail_right = example
                train_examples.append(
                    InputExample(guid='train', text_a=sentence, span_a_left=head_left, span_a_right=head_right, text_b=None, span_b_left=tail_left, span_b_right=tail_right, label=relation))

        '''build dev'''
        dev_examples = []
        for relation, example_list in dev_4_dev.items():
            for example in example_list:
                sentence, head_left, head_right, tail_left, tail_right = example
                dev_examples.append(
                    InputExample(guid='dev', text_a=sentence, span_a_left=head_left, span_a_right=head_right, text_b=None, span_b_left=tail_left, span_b_right=tail_right, label=relation))

        # print('dev_examples:', len(dev_examples))
        '''build test'''
        test_examples = []
        for relation, example_list in dev_4_test.items():
            for example in example_list:
                sentence, head_left, head_right, tail_left, tail_right = example
                test_examples.append(
                    InputExample(guid='test', text_a=sentence, span_a_left=head_left, span_a_right=head_right, text_b=None, span_b_left=tail_left, span_b_right=tail_right, label=relation))



        return train_examples, dev_examples, test_examples, relation_list

    def get_labels(self):
        'here we keep the three-way in MNLI training '
        return ["entailment", "not_entailment"]
        # return ["entailment", "neutral", "contradiction"]



def wordpairID_2_tokenpairID(sentence, wordindex_left, wordindex_right, full_token_id_list, tokenizer, sent_1=True):
    '''pls note that the input indices pair include the b in (a,b), but the output doesn't'''
    '''first find the position of [2,2]'''
    position_two_two = 0
    for i in range(len(full_token_id_list)):
        if full_token_id_list[i]==2 and full_token_id_list[i+1]==2:
            position_two_two = i
            break
    span = ' '.join(sentence.split()[wordindex_left: wordindex_right+1])
    if wordindex_left!=0:
        '''this span is the begining of the sent'''
        span=' '+span

    span_token_list = tokenizer.tokenize(span)
    span_id_list = tokenizer.convert_tokens_to_ids(span_token_list)
    # print('span:', span, 'span_id_list:', span_id_list)
    if sent_1:
        # for i in range(wordindex_left, len(full_token_id_list)-len(span_id_list)):
        for i in range(wordindex_left, position_two_two):
            if full_token_id_list[i:i+len(span_id_list)] == span_id_list:
                return i, i+len(span_id_list), span_token_list

        return None, None, span_token_list
    else:
        # print('position_two_two:', position_two_two)
        for i in range(position_two_two+2, len(full_token_id_list)):
            if full_token_id_list[i:i+len(span_id_list)] == span_id_list:
                return i, i+len(span_id_list), span_token_list

        return None, None, span_token_list


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    give_up = 0
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)
        # print('tokens_a:', tokens_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # print('tokens_b:', tokens_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
            special_tokens_count = 4 if sep_token_extra else 3
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
        else:
            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2
            if len(tokens_a) > max_seq_length - special_tokens_count:
                tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]

        tokens = tokens_a + [sep_token]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)


        span_a_left, span_a_right, span_a_token_list = wordpairID_2_tokenpairID(example.text_a, example.span_a_left, example.span_a_right, input_ids, tokenizer, sent_1=True)
        span_b_left, span_b_right, span_b_token_list = wordpairID_2_tokenpairID(example.text_a, example.span_b_left, example.span_b_right, input_ids, tokenizer, sent_1=True)
        # print('span_b_left, span_b_right, span_b_token_list:', span_b_left, span_b_right, span_b_token_list)
        if span_a_left is None or span_b_left is None:
            '''give up this pair'''
            give_up+=1
            continue
        else:
            span_a_mask = [0]*len(input_ids)
            for i in range(span_a_left, span_a_right):
                span_a_mask[i]=1
            span_b_mask = [0]*len(input_ids)
            for i in range(span_b_left, span_b_right):
                span_b_mask[i]=1
            features.append(
                    InputFeatures(input_ids=input_ids,
                                  input_mask=input_mask,
                                  segment_ids=segment_ids,
                                  span_a_mask = span_a_mask,
                                  span_b_mask = span_b_mask,
                                  label_id=label_id))
    print('input example size:', len(examples), ' give_up:', give_up, ' remain:', len(features))
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










def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    ## Other parameters
    parser.add_argument('--batch_mix_times',
                        type=int,
                        default=10,
                        help="random seed for initialization")
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")

    parser.add_argument('--kshot',
                        type=float,
                        default=5,
                        help="random seed for initialization")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=1e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--use_mixup",
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
    parser.add_argument('--beta_sampling_times',
                        type=int,
                        default=10,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")


    args = parser.parse_args()


    processors = {
        "rte": RteProcessor
    }

    output_modes = {
        "rte": "classification"
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")


    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))



    processor = processors[task_name]()
    output_mode = output_modes[task_name]



    train_examples, dev_examples, test_examples, label_list = processor.load_FewRel_data(args.kshot)

    num_labels = len(label_list)
    print('num_labels:', num_labels, 'training size:', len(train_examples), 'dev size:', len(dev_examples), 'test size:', len(test_examples))


    model = RobertaForSequenceClassification(num_labels)
    tokenizer = RobertaTokenizer.from_pretrained(pretrain_model_dir, do_lower_case=args.do_lower_case)
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters,
                             lr=args.learning_rate)
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    max_test_acc = 0.0
    max_dev_acc = 0.0
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, output_mode,
            cls_token_at_end=False,#bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=0,#2 if args.model_type in ['xlnet'] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=True,#bool(args.model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=False,#bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=0)#4 if args.model_type in ['xlnet'] else 0,)

        '''load dev set'''
        dev_features = convert_examples_to_features(
            dev_examples, label_list, args.max_seq_length, tokenizer, output_mode,
            cls_token_at_end=False,#bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=0,#2 if args.model_type in ['xlnet'] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=True,#bool(args.model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=False,#bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=0)#4 if args.model_type in ['xlnet'] else 0,)

        dev_all_input_ids = torch.tensor([f.input_ids for f in dev_features], dtype=torch.long)
        dev_all_input_mask = torch.tensor([f.input_mask for f in dev_features], dtype=torch.long)
        dev_all_segment_ids = torch.tensor([f.segment_ids for f in dev_features], dtype=torch.long)
        dev_all_span_a_mask = torch.tensor([f.span_a_mask for f in dev_features], dtype=torch.float)
        dev_all_span_b_mask = torch.tensor([f.span_b_mask for f in dev_features], dtype=torch.float)

        dev_all_label_ids = torch.tensor([f.label_id for f in dev_features], dtype=torch.long)

        dev_data = TensorDataset(dev_all_input_ids, dev_all_input_mask, dev_all_segment_ids, dev_all_span_a_mask, dev_all_span_b_mask, dev_all_label_ids)
        dev_sampler = SequentialSampler(dev_data)
        dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.eval_batch_size)


        '''load test set'''
        test_features = convert_examples_to_features(
            test_examples, label_list, args.max_seq_length, tokenizer, output_mode,
            cls_token_at_end=False,#bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=0,#2 if args.model_type in ['xlnet'] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=True,#bool(args.model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=False,#bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=0)#4 if args.model_type in ['xlnet'] else 0,)

        eval_all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        eval_all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        eval_all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        eval_all_span_a_mask = torch.tensor([f.span_a_mask for f in test_features], dtype=torch.float)
        eval_all_span_b_mask = torch.tensor([f.span_b_mask for f in test_features], dtype=torch.float)
        # eval_all_pair_ids = [f.pair_id for f in test_features]
        eval_all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)

        eval_data = TensorDataset(eval_all_input_ids, eval_all_input_mask, eval_all_segment_ids, eval_all_span_a_mask, eval_all_span_b_mask, eval_all_label_ids)
        eval_sampler = SequentialSampler(eval_data)
        test_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size)
        # logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_span_a_mask = torch.tensor([f.span_a_mask for f in train_features], dtype=torch.float)
        all_span_b_mask = torch.tensor([f.span_b_mask for f in train_features], dtype=torch.float)

        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_span_a_mask, all_span_b_mask, all_label_ids)
        train_sampler = RandomSampler(train_data)

        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        iter_co = 0
        final_test_performance = 0.0
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                model.train()
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, span_a_mask, span_b_mask, label_ids = batch
                real_batch_size = input_ids.shape[0]
                lambda_vec = torch.rand(args.batch_mix_times, real_batch_size).to(device)
                softmax_lambda_vec = nn.Softmax(dim=1)(lambda_vec) #(mix_time, batch_size)

                logits = model(input_ids, input_mask, span_a_mask, span_b_mask, softmax_lambda_vec, is_train=args.use_mixup)
                loss_fct = CrossEntropyLoss(reduction='none')
                if args.use_mixup:
                    '''mixup loss'''
                    mixup_logits = logits.view(-1, num_labels) #(mixup_times, 2)
                    mixup_logits_repeat = tile(mixup_logits, 0, real_batch_size)
                    label_id_repeat = label_ids.view(-1).repeat(args.batch_mix_times) #(0,1,2,..batch, 0, 1,2,3...batch)
                    mixup_loss_repeat = loss_fct(mixup_logits_repeat.view(-1, num_labels), label_id_repeat.view(-1))
                    mixup_loss = torch.sum(mixup_loss_repeat.view(args.batch_mix_times, real_batch_size)*softmax_lambda_vec, dim=1) #(mixup_time)
                    loss = mixup_loss.mean()

                else:
                    loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.

                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                iter_co+=1
                # if iter_co %20==0:
                if iter_co % len(train_dataloader)==0:
                    # if iter_co % (len(train_dataloader)//2)==0:
                    '''
                    start evaluate on dev set after this epoch
                    '''
                    model.eval()

                    for idd, dev_or_test_dataloader in enumerate([dev_dataloader, test_dataloader]):


                        if idd == 0:
                            logger.info("***** Running dev *****")
                            logger.info("  Num examples = %d", len(dev_features))
                        else:
                            logger.info("***** Running test *****")
                            logger.info("  Num examples = %d", len(test_features))
                        # logger.info("  Batch size = %d", args.eval_batch_size)

                        eval_loss = 0
                        nb_eval_steps = 0
                        preds = []
                        gold_label_ids = []
                        # print('Evaluating...')
                        for input_ids, input_mask, segment_ids, span_a_mask, span_b_mask, label_ids in dev_or_test_dataloader:
                            input_ids = input_ids.to(device)
                            input_mask = input_mask.to(device)
                            segment_ids = segment_ids.to(device)
                            span_a_mask = span_a_mask.to(device)
                            span_b_mask = span_b_mask.to(device)
                            label_ids = label_ids.to(device)
                            gold_label_ids+=list(label_ids.detach().cpu().numpy())

                            with torch.no_grad():
                                logits = model(input_ids, input_mask, span_a_mask, span_b_mask, None, is_train=False)
                            if len(preds) == 0:
                                preds.append(logits.detach().cpu().numpy())
                            else:
                                preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)

                        preds = preds[0]

                        pred_probs = softmax(preds,axis=1)
                        pred_label_ids = list(np.argmax(pred_probs, axis=1))

                        assert len(pred_label_ids) == len(gold_label_ids)
                        hit_co = 0
                        for k in range(len(pred_label_ids)):
                            if pred_label_ids[k] == gold_label_ids[k]:
                                hit_co +=1
                        test_acc = hit_co/len(gold_label_ids)
                        f1 = test_acc


                        if idd == 0: # this is dev
                            if f1 > max_dev_acc:
                                max_dev_acc = f1
                                print('\ndev acc :', f1, ' max_dev_acc:', max_dev_acc, '\n')
                                '''store the model, because we can test after a max_dev acc reached'''
                                model_to_save = (
                                    model.module if hasattr(model, "module") else model
                                )  # Take care of distributed/parallel training
                                store_transformers_models(model_to_save, tokenizer, '/export/home/Dataset/BERT_pretrained_mine/mixup_wenpeng', 'FewRel_batchMixup_pretrain_kshot_'+str(args.kshot)+'_dev_acc_seed_'+str(args.seed)+'.pt')

                            else:
                                print('\ndev acc :', f1, ' max_dev_acc:', max_dev_acc, '\n')
                                break
                        else: # this is test
                            if f1 > max_test_acc:
                                max_test_acc = f1
                            final_test_performance = f1
                            print('\ntest acc:', f1, ' max_test_acc:', max_test_acc, '\n')
        print('final_test_f1:', final_test_performance)


if __name__ == "__main__":
    main()

'''

CUDA_VISIBLE_DEVICES=1 python -u train.FewRel.batchMixup.pretrain.py --task_name rte --do_train --do_lower_case --num_train_epochs 10  --train_batch_size 32 --eval_batch_size 128 --learning_rate 1e-5 --max_seq_length 64 --seed 42 --kshot 0 --use_mixup --batch_mix_times 400


'''
