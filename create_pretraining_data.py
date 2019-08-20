# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""Create masked LM/next sentence masked_lm TF examples for BERT."""
from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import argparse
import logging
import os
import random
from io import open
import h5py
import numpy as np
from tqdm import tqdm, trange
import json

from tokenization import BertTokenizer
import tokenization as tokenization

import random
import collections
import mask_utils.mask_generators as mask_generators



class TrainingInstance(object):
    """A single training instance (sentence pair)."""
    def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels, is_random_next):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.is_random_next = is_random_next
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.tokens]))
        s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
        s += "is_random_next: %s\n" % self.is_random_next
        s += "masked_lm_positions: %s\n" % (" ".join(
            [str(x) for x in self.masked_lm_positions]))
        s += "masked_lm_labels: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.masked_lm_labels]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


def write_instance_to_example_file(instances, tokenizer, max_seq_length,
                                    max_predictions_per_seq, output_file):
    """Create TF example files from `TrainingInstance`s."""
    
    total_written = 0
    features = collections.OrderedDict()
    
    num_instances = len(instances)
    features["input_ids"] = np.zeros([num_instances, max_seq_length], dtype="int32")
    features["input_mask"] = np.zeros([num_instances, max_seq_length], dtype="int32")
    features["segment_ids"] = np.zeros([num_instances, max_seq_length], dtype="int32")
    features["masked_lm_positions"] = np.zeros([num_instances, max_predictions_per_seq], dtype="int32")
    features["masked_lm_ids"] = np.zeros([num_instances, max_predictions_per_seq], dtype="int32")
    features["next_sentence_labels"] = np.zeros(num_instances, dtype="int32")


    for inst_index, instance in enumerate(tqdm(instances)):
        input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = list(instance.segment_ids)
        assert len(input_ids) <= max_seq_length

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        masked_lm_positions = list(instance.masked_lm_positions)
        masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
        masked_lm_weights = [1.0] * len(masked_lm_ids)

        while len(masked_lm_positions) < max_predictions_per_seq:
            masked_lm_positions.append(0)
            masked_lm_ids.append(0)
            masked_lm_weights.append(0.0)

        next_sentence_label = 1 if instance.is_random_next else 0



        features["input_ids"][inst_index] = input_ids
        features["input_mask"][inst_index] = input_mask
        features["segment_ids"][inst_index] = segment_ids
        features["masked_lm_positions"][inst_index] = masked_lm_positions
        features["masked_lm_ids"][inst_index] = masked_lm_ids
        features["next_sentence_labels"][inst_index] = next_sentence_label

        total_written += 1

        # if inst_index < 20:
        #   tf.logging.info("*** Example ***")
        #   tf.logging.info("tokens: %s" % " ".join(
        #       [tokenization.printable_text(x) for x in instance.tokens])      
        #   for feature_name in features.keys():
        #     feature = features[feature_name]
        #     values = []
        #     if feature.int64_list.value:
        #       values = feature.int64_list.value
        #     elif feature.float_list.value:
        #       values = feature.float_list.value
        #     tf.logging.info(
        #         "%s: %s" % (feature_name, " ".join([str(x) for x in values])))


    print("saving data")
    f= h5py.File(output_file, 'w')
    f.create_dataset("input_ids", data=features["input_ids"], dtype='i4', compression='gzip')
    f.create_dataset("input_mask", data=features["input_mask"], dtype='i1', compression='gzip')
    f.create_dataset("segment_ids", data=features["segment_ids"], dtype='i1', compression='gzip')
    f.create_dataset("masked_lm_positions", data=features["masked_lm_positions"], dtype='i4', compression='gzip')
    f.create_dataset("masked_lm_ids", data=features["masked_lm_ids"], dtype='i4', compression='gzip')
    f.create_dataset("next_sentence_labels", data=features["next_sentence_labels"], dtype='i1', compression='gzip')
    f.flush()
    f.close()

def tokenize(tokenizer, line):
    words = line.strip().split(" ")
    tokens = []
    valid_positions = []
    for i, word in enumerate(words):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        for i in range(len(token)):
            if i == 0:
                valid_positions.append(1)
            else:
                valid_positions.append(0)
    return tokens, valid_positions

def create_better_mask(task_name, signi_indexes, tokens, valid_positions, masked_lm_prob, max_predictions_rate, vocab_words, rng):
    """Creates the predictions for the masked LM objective."""
    # NOTE this sequence is defined as the sequence after concatenation after devided by 2, the final mask number should be OK
    max_predictions_sub_seq = len(tokens) * max_predictions_rate
    cand_indexes = []
    # save all word parts tokenized from a single word in a list
    if task_name:
        for index in signi_indexes:
            cand_indexes.append([index])
            i = index + 1
            while i < len(valid_positions) and valid_positions[i] == 0:
                cand_indexes[-1].append(i)
                i += 1
    else:
        # using default strategy to choose mask positions
        # randomly shuffle all indices
        for (i, valid) in enumerate(valid_positions):
            if valid == 1:
                cand_indexes.append([i])
            else:
                cand_indexes[-1].append(i)
        rng.shuffle(cand_indexes)

    # output_tokens = list(tokens)

    # NOTE changed to len(cand_indexes) * masked_lm_prob
    num_to_predict = min(max_predictions_sub_seq, max(1, int(round(len(cand_indexes) * masked_lm_prob))))

    masked_info = [{} for token in tokens] # if masked, masked symbol, else ""
    masked_lms_len = 0
    # covered_indexes = set()
    for indexes in cand_indexes:
        if masked_lms_len >= num_to_predict:
            break
        # if index in covered_indexes:
        #   continue
        # covered_indexes.add(index)

        masked_tokens = None
        # 80% of the time, replace with [MASK]
        if rng.random() < 0.8:
            masked_tokens = ["[MASK]" for index in indexes]
        else:
            # 10% of the time, keep original
            if rng.random() < 0.5:
                masked_tokens = [tokens[index] for index in indexes]
            # 10% of the time, replace with random word
            else:
                masked_tokens = [vocab_words[rng.randint(0, len(vocab_words) - 1)] for index in indexes]

        for (i, index) in enumerate(indexes):
            masked_info[index]["mask"] = masked_tokens[i]
            masked_info[index]["label"] = tokens[index]

        masked_lms_len += len(indexes)

        # masked_lms.extend([MaskedLmInstance(index=index, label=tokens[index]) for index in indexes])

    # masked_lms = sorted(masked_lms, key=lambda x: x.index)

    # masked_lm_positions = []
    # masked_lm_labels = []
    # for p in masked_lms:
        # masked_lm_positions.append(p.index)
        # masked_lm_labels.append(p.label)

    return masked_info

def create_training_instances(input_files, task_name, generator, tokenizer, max_seq_length, dupe_factor, short_seq_prob, masked_lm_prob, max_predictions_per_seq, rng):
    """Create `TrainingInstance`s from raw text."""
    all_documents = [[]]
    vocab_words = list(tokenizer.vocab.keys())

    # Input file format:
    # (1) One sentence per line. These should ideally be actual sentences, not
    # entire paragraphs or arbitrary spans of text. (Because we use the
    # sentence boundaries for the "next sentence prediction" task).
    # (2) Blank lines between documents. Document boundaries are needed so
    # that the "next sentence prediction" task doesn't span between documents.
    for input_file in input_files:
        print("creating instance from {}".format(input_file))
        with open(input_file, "r") as reader:
            lines = reader.readlines()
            i = 0
            for line in tqdm(lines[76:78], desc="Processing"):
                line = tokenization.convert_to_unicode(line)
                line = line.strip()

                # Empty lines are used as document delimiters
                if not line:
                    i += 1
                    all_documents.append([])
                    continue

                # tokenize
                # tokens = tokenizer.tokenize(line)
                signi_indexes = []
                if task_name:
                    signi_indexes = generator(line)

                tokens, valid_positions = tokenize(tokenizer, line)
                m_info = create_better_mask(task_name, signi_indexes, tokens, valid_positions, masked_lm_prob, max_predictions_per_seq / max_seq_length, vocab_words, rng)
                if tokens:
                    all_documents[-1].append(MaskedTokenInstance(tokens=tokens, info=m_info))

    # Remove empty documents
    all_documents = [x for x in all_documents if x]
    # rng.shuffle(all_documents)

    instances = []
    for _ in range(dupe_factor):
        for document_index in range(len(all_documents)):
            instances.extend(create_instances_from_document(all_documents, document_index, max_seq_length, short_seq_prob,
                masked_lm_prob, max_predictions_per_seq, vocab_words, rng))

    rng.shuffle(instances)
    return instances


def create_instances_from_document(
    all_documents, document_index, max_seq_length, short_seq_prob,
    masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
    """Creates `TrainingInstance`s for a single document."""

    # document: MaskedTokenInstance: (tokens, info)
    document = all_documents[document_index]

    # Account for [CLS], [SEP]
    max_num_tokens = max_seq_length - 2

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if rng.random() < short_seq_prob:
        target_seq_length = rng.randint(2, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i] # segment: MaskedTokenInstance (tokens, info)
        current_chunk.append(segment)
        current_length += len(segment.tokens)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                # a_end = 1
                # if len(current_chunk) >= 2:
                #   a_end = rng.randint(1, len(current_chunk) - 1)

                tokens_a = []
                m_info_a = []
                # for j in range(a_end):
                for j in range(len(current_chunk)):
                    tokens_a.extend(current_chunk[j].tokens)
                    m_info_a.extend(current_chunk[j].info)
                truncate_seq_pair(tokens_a, m_info_a, [], [], max_num_tokens, rng)

                # tokens_b = []
                # m_info_b = []
                # Random next
                # is_random_next = False
                # if len(current_chunk) == 1 or rng.random() < 0.5:
                #     is_random_next = True
                #     target_b_length = target_seq_length - len(tokens_a)

                #     # This should rarely go for more than one iteration for large
                #     # corpora. However, just to be careful, we try to make sure that
                #     # the random document is not the same as the document
                #     # we're processing.
                #     for _ in range(10):
                #       random_document_index = rng.randint(0, len(all_documents) - 1)
                #       if random_document_index != document_index:
                #         break

                #     random_document = all_documents[random_document_index]
                #     random_start = rng.randint(0, len(random_document) - 1)
                #     for j in range(random_start, len(random_document)):
                #       tokens_b.extend(random_document[j].tokens)
                #       m_info_b.extend(random_document[j].info)
                #       if len(tokens_b) >= target_b_length:
                #         break
                #     # We didn't actually use these segments so we "put them back" so
                #     # they don't go to waste.
                #     num_unused_segments = len(current_chunk) - a_end
                #     i -= num_unused_segments
                # # Actual next
                # else:
                #     is_random_next = False
                #     for j in range(a_end, len(current_chunk)):
                #         tokens_b.extend(current_chunk[j].tokens)
                #         m_info_b.extend(current_chunk[j].info)


                assert len(tokens_a) >= 1
                # assert len(tokens_b) >= 1

                tokens = []
                m_info = []
                segment_ids = []
                tokens.append("[CLS]")
                m_info.append({})
                segment_ids.append(0)
                for token, info in zip(tokens_a, m_info_a):
                    tokens.append(token)
                    m_info.append(info)
                    segment_ids.append(0)

                tokens.append("[SEP]")
                m_info.append({})
                segment_ids.append(0)

                # for token, info in zip(tokens_b, m_info_b):
                #     tokens.append(token)
                #     m_info.append(info)
                #     segment_ids.append(1)
                # tokens.append("[SEP]")
                # m_info.append("")
                # segment_ids.append(1)

                masked_lm_positions = [index for index in range(len(m_info)) if m_info[index]]
                if len(masked_lm_positions) > max_predictions_per_seq:
                    rng.shuffle(masked_lm_positions)
                    masked_lm_positions = masked_lm_positions[0:max_predictions_per_seq]
                    masked_lm_positions.sort()
                # masks = [m_info[pos]["mask"] for pos in masked_lm_positions]
                masked_lm_labels = [m_info[pos]["label"] for pos in masked_lm_positions]
                
                for pos in masked_lm_positions:
                    tokens[pos] = m_info[pos]["mask"]

                # for (pos, label) in zip(masked_lm_positions, masks):
                    # tokens[pos] = label                   
                
                # (tokens, masked_lm_positions, masked_lm_labels) = create_masked_lm_predictions(
                    #  tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
                is_random_next = False
                instance = TrainingInstance(
                    tokens=tokens,
                    segment_ids=segment_ids,
                    is_random_next=is_random_next,
                    masked_lm_positions=masked_lm_positions,
                    masked_lm_labels=masked_lm_labels)
                instances.append(instance)
                # print(tokens, masked_lm_positions, masked_lm_labels)
            current_chunk = []
            current_length = 0  
        i += 1
    return instances


MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])
MaskedTokenInstance = collections.namedtuple("MaskedTokenInstance", ["tokens", "info"])

def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng):
    """Creates the predictions for the masked LM objective."""

    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        cand_indexes.append(i)

    rng.shuffle(cand_indexes)

    output_tokens = list(tokens)

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []
    # covered_indexes = set()
    for index in cand_indexes:
        if len(masked_lms) >= num_to_predict:
          break
        # if index in covered_indexes:
        #   continue
        # covered_indexes.add(index)

        masked_token = None
        # 80% of the time, replace with [MASK]
        if rng.random() < 0.8:
          masked_token = "[MASK]"
        else:
          # 10% of the time, keep original
          if rng.random() < 0.5:
            masked_token = tokens[index]
          # 10% of the time, replace with random word
          else:
            masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

        output_tokens[index] = masked_token

        masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)
    return (output_tokens, masked_lm_positions, masked_lm_labels)

def truncate_seq_pair(tokens_a, m_info_a, tokens_b, m_info_b, max_num_tokens, rng):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        (trunc_tokens, trunc_info) = (tokens_a, m_info_a) if len(tokens_a) > len(tokens_b) else (tokens_b, m_info_b)
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if rng.random() < 0.5:
            del trunc_tokens[0]
            del trunc_info[0]
        else:
            trunc_tokens.pop()
            trunc_info.pop()


def main():

    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--vocab_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The vocabulary the BERT model will train on.")
    parser.add_argument("--input_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input train corpus. can be directory with .txt files or a path to a single file")
    parser.add_argument("--input_prefix",
                        default=None,
                        type=str,
                        )
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output file where the model checkpoints will be written.")

    ## Other parameters

    # str
    parser.add_argument("--bert_model", 
                        default="bert-large-uncased", 
                        type=str, 
                        required=False,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                              "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--task_name", 
                        default="", 
                        type=str,
                        required=False,
                        help="Use specific task to generate better mask. If does not specify a task, "
                                "mask will be randomly chose as original version")
    parser.add_argument("--downstream_config", 
                        default="",
                        type=str,
                        required=False,
                        help="Downstream model configure json file")
    parser.add_argument("--gpus", 
                        default=0,
                        type=int)
    parser.add_argument("--local_rank",
                        default=0,
                        type=int)

    # int 
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--dupe_factor",
                        default=10,
                        type=int,
                        help="Number of times to duplicate the input data (with different masks).")
    parser.add_argument("--max_predictions_per_seq",
                        default=20,
                        type=int,
                        help="Maximum sequence length.")
                             

    # floats

    parser.add_argument("--masked_lm_prob",
                        default=0.15,
                        type=float,
                        help="Masked LM probability.")

    parser.add_argument("--short_seq_prob",
                        default=0.1,
                        type=float,
                        help="Probability to create a sequence shorter than maximum sequence length")

    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument('--random_seed',
                        type=int,
                        default=12345,
                        help="random seed for initialization")

    args = parser.parse_args()
    print(args)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    generator = None
    if args.task_name:
        downstream_config = json.load(open(args.downstream_config))[
            args.task_name]
        generator = getattr(mask_generators, args.task_name)(downstream_config)
    rng = random.Random(args.random_seed)

    # input_files = []
    rank = args.local_rank
    filename = os.path.join(args.input_dir, args.input_prefix + str(rank) + ".txt")
    k = 0
    while os.path.isfile(filename):
        input_files = [filename]
        
        # if os.path.isfile(args.input_file):
        # input_files.append(args.input_file)
        # elif os.path.isdir(args.input_file):
        # input_files = [os.path.join(args.input_file, f) for f in os.listdir(args.input_file) if (os.path.isfile(os.path.join(args.input_file, f)) and f.endswith('.txt'))]
        # else:
        # raise ValueError("{} is not a valid path".format(args.input_file))
        
        # print(args)
        # torch.cuda.set_device(args.local_rank)
        instances = create_training_instances(
            input_files, args.task_name, generator, tokenizer, args.max_seq_length, args.dupe_factor,
            args.short_seq_prob, args.masked_lm_prob, args.max_predictions_per_seq,
            rng)
            
        output_file = os.path.join(args.output_dir, str(rank) + ".hdf5")        
        write_instance_to_example_file(instances, tokenizer, args.max_seq_length, args.max_predictions_per_seq, output_file)
        
        rank += args.gpus
        k += 1
        filename = os.path.join(args.input_dir, args.input_prefix + str(rank) + ".txt")


if __name__ == "__main__":
    main()
