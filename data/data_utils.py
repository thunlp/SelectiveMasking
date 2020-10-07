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
""" BERT classification fine-tuning: utilities to work with GLUE tasks """

from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
import xml.etree.ElementTree as ET

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
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
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
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
    def _read_tsv(cls, input_file, quotechar=None, delimiter="\t"):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(cell for cell in line)
                lines.append(line)
            return lines


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")),
            "dev_matched")


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class ABSAProcessor(DataProcessor):
    def _create_absa_examples(self, L, set_type):
        examples = []
        for (i, line) in enumerate(L):
            guid = "%s-%s" % (set_type, i)
            text_b = line["text"]
            for label in line["labels"]:
                if label["polarity"] != "conflict":
                    examples.append(InputExample(guid=guid, text_a=label["category"], text_b=text_b, label=label["polarity"]))
        return examples
    
    def get_train_examples(self, data_dir):
        return self._create_absa_examples(self._read_xml(os.path.join(data_dir, "train.xml")), "train")
    
    def get_dev_examples(self, data_dir):
        return self._create_absa_examples(self._read_xml(os.path.join(data_dir, "dev.xml")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_absa_examples(self._read_xml(os.path.join(data_dir, "test.xml")), "test")

    def get_pretrain_examples(self, data_dir, part, max_proc):
        """See base class"""
        lines = self._read_xml(os.path.join(data_dir, "train.xml"))
        data_size = len(lines)
        part_size = data_size // max_proc
        begin = 0
        end = data_size
        if part >= 0:
            begin = part*part_size
            end = (part+1)*part_size
            # print(begin, end)
            # if end - begin < 1000:
            # begin = 0
            # end = data_size
            if part == max_proc - 1:
                end = data_size
        examples = []
        for i in range(begin, end):
            line = lines[i]
            examples.append({"text": line["text"], "facts": line["labels"]})
        return examples

    def _read_xml(self, path):
        elements = ET.parse(path).getroot().findall("sentence")
        L = []
        for e in elements:
            text = e.find("text").text
            labels = []
            for et in e.findall("aspectCategories"):
                for es in et:
                    labels.append({"category": es.attrib["category"], "polarity": es.attrib["polarity"]})    
            L.append({"text": text, "labels": labels})

        return L

    def get_labels(self):
        return ["positive", "negative", "neutral"]

class ABSATermProcessor(DataProcessor):
    def _create_absa_examples(self, L, set_type):
        examples = []
        for (i, line) in enumerate(L):
            guid = "%s-%s" % (set_type, i)
            text_b = line["text"]
            for label in line["labels"]:
                if label["polarity"] != "conflict":
                    examples.append(InputExample(guid=guid, text_a=label["term"], text_b=text_b, label=label["polarity"]))
        return examples
    
    def get_train_examples(self, data_dir):
        return self._create_absa_examples(self._read_xml(os.path.join(data_dir, "train.xml")), "train")
    
    def get_dev_examples(self, data_dir):
        return self._create_absa_examples(self._read_xml(os.path.join(data_dir, "dev.xml")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_absa_examples(self._read_xml(os.path.join(data_dir, "test.xml")), "test")

    def get_pretrain_examples(self, data_dir, part, max_proc):
        """See base class"""
        lines = self._read_xml(os.path.join(data_dir, "train.xml"))
        data_size = len(lines)
        print(data_size)
        part_size = data_size // max_proc
        begin = 0
        end = data_size
        if part >= 0:
            begin = part*part_size
            end = (part+1)*part_size
            # print(begin, end)
            # if end - begin < 1000:
            # begin = 0
            # end = data_size
            if part == max_proc - 1:
                end = data_size
        print(part, max_proc, begin, end)
        examples = []
        for i in range(begin, end):
            line = lines[i]
            examples.append({"text": line["text"], "facts": line["labels"]})
        return examples

    def _read_xml(self, path):
        elements = ET.parse(path).getroot().findall("sentence")
        L = []
        for e in elements:
            text = e.find("text").text
            labels = []
            for et in e.findall("aspectTerms"):
                for es in et:
                    labels.append({"term": es.attrib["term"], "polarity": es.attrib["polarity"]})    
            L.append({"text": text, "labels": labels})

        return L

    def get_labels(self):
        return ["positive", "negative", "neutral"]

class TwitterProcessor(DataProcessor):
    """Processor for the Twitter dataset"""
    def get_train_examples(self, data_dir):
        print("get train examples")
        return self._create_examples(self._read_twitter(os.path.join(data_dir, "train.tsv")), "train")
    
    def get_dev_examples(self, data_dir):
        print("get dev examples")
        return self._create_examples(self._read_twitter(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        print("get test examples")
        return self._create_examples(self._read_twitter(os.path.join(data_dir, "test.tsv")), "test")

    def get_pretrain_examples(self, data_dir, part, max_proc):
        """See base class"""
        lines = self._read_twitter(os.path.join(data_dir, "train.tsv"))
        data_size = len(lines)
        part_size = data_size // max_proc
        begin = 0
        end = data_size
        if part >= 0:
            begin = part*part_size
            end = (part+1)*part_size
            # print(begin, end)
            # if end - begin < 1000:
                # begin = 0
                # end = data_size
            if part == max_proc - 1:
                end = data_size
        examples = []
        for i in range(begin, end):
            line = lines[i]
            if line[1] == "positive":
                label = '2'
            elif line[1] == "negative":
                label = '1'
            elif line[1] == "neutral":
                label = '0'
            else:
                assert(0 == 1)
            text_a = line[2]
            print(text_a, label)
            examples.append(InputExample(guid=i, text_a=text_a, text_b=None, label=label))
        return examples


    def _read_twitter(self, input_file):
        L = []
        with open(input_file, 'r') as f:
            for line in f:
                L.append(line.strip().split("\t"))
        return L

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[2]
            if line[1] == "positive":
                label = '2'
            elif line[1] == "neutral":
                label = '1'
            elif line[1] == "negative":
                label = '0'
            else:
                assert(0 == 1)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def get_labels(self):
        return ['0', '1', '2']

    

class MRProcessor(DataProcessor):
    """Processor for the MR dataset"""
    def get_train_examples(self, data_dir):
        """See base class"""
        print("get train examples")
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.csv"), quotechar='*', delimiter=','), "train")

    def get_pretrain_examples(self, data_dir, part, max_proc):
        """See base class"""
        lines = self._read_tsv(os.path.join(data_dir, "train.csv"), quotechar='*', delimiter=',')
        data_size = len(lines)
        part_size = data_size // max_proc
        begin = 0
        end = data_size
        if part >= 0:
            begin = part*part_size
            end = (part+1)*part_size
            # print(begin, end)
            # if end - begin < 1000:
                # begin = 0
                # end = data_size
            if part == max_proc - 1:
                end = data_size
        examples = []
        for i in range(begin, end):
            line = lines[i]
            label = line[0]
            text_a = line[1]
            examples.append(InputExample(guid=i, text_a=text_a, text_b=None, label=label))
        return examples


    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.csv"), quotechar='*', delimiter=','), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.csv"), quotechar='*', delimiter=','), "test")

    def get_labels(self):
        """See base class."""
        return ['0', '1']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            label = line[0]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples
        
class YelpProcessor(DataProcessor):
    """Processor for the Yelp data set"""
    def get_train_examples(self, data_dir):
        """See base class"""
        print("get train examples")
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.csv"), quotechar='"', delimiter=','), "train")

    def get_pretrain_examples(self, data_dir, part, max_proc):
        """See base class"""
        print(part)
        print(max_proc)
        lines = self._read_tsv(os.path.join(data_dir, "train.csv"), quotechar='"', delimiter=',')
        data_size = len(lines)
        part_size = data_size // max_proc
        begin = 0
        end = data_size
        if part >= 0:
            begin = part*part_size
            end = (part+1)*part_size
            # print(begin, end)
            # if end - begin < 1000:
                # begin = 0
                # end = data_size
            if part == max_proc - 1:
                end = data_size
        examples = []
        for i, line in enumerate(lines[begin:end]):
            label = line[0]
            text_a = line[1]
            examples.append(InputExample(guid=i, text_a=text_a, text_b=None, label=label))
        return examples


    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.csv"), quotechar='"', delimiter=','), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.csv"), quotechar='*', delimiter=','), "test")

    def get_labels(self):
        """See base class."""
        return ['1', '2', '3', '4', '5']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            label = line[0]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class AmazonProcessor(DataProcessor):
    """Processor for the Yelp data set"""
    def get_train_examples(self, data_dir):
        """See base class"""
        print("get train examples")
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.csv"), quotechar='"', delimiter=','), "train")

    def get_pretrain_examples(self, data_dir, part, max_proc):
        """See base class"""
        print(part)
        print(max_proc)
        lines = self._read_tsv(os.path.join(data_dir, "train.csv"), quotechar='"', delimiter=',')
        data_size = len(lines)
        part_size = data_size // max_proc
        begin = 0
        end = data_size
        if part >= 0:
            begin = part*part_size
            end = (part+1)*part_size
            # print(begin, end)
            # if end - begin < 1000:
                # begin = 0
                # end = data_size
            if part == max_proc - 1:
                end = data_size
        examples = []
        for i, line in enumerate(lines[begin:end]):
            label = line[0]
            text_a = line[1] + ". " + line[2]
            examples.append(InputExample(guid=i, text_a=text_a, text_b=None, label=label))
        return examples


    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.csv"), quotechar='"', delimiter=','), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.csv"), quotechar='*', delimiter=','), "test")

    def get_labels(self):
        """See base class."""
        return ['1', '2', '3', '4', '5']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[1] + ". " + line[2]
            label = line[0]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[3]
                text_b = line[4]
                label = line[5]
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), 
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        # padding = [0] * (max_seq_length - len(input_ids))
        # input_ids += padding
        # input_mask += padding
        # segment_ids += padding

        # assert len(input_ids) == max_seq_length
        # assert len(input_mask) == max_seq_length
        # assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_id=label_id))
        
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


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1_micro = f1_score(y_true=labels, y_pred=preds, average="micro")
    f1_macro = f1_score(y_true=labels, y_pred=preds, average="macro")
    return {
        "acc": acc,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        # "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "yelp":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "amazon":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mr":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "twitter":
        return acc_and_f1(preds, labels)
    elif task_name == "absa":
        return acc_and_f1(preds, labels)
    elif task_name == "absa_term":
        return acc_and_f1(preds, labels)
    else:
        raise KeyError(task_name)

processors = {
    "cola": ColaProcessor,
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "mrpc": MrpcProcessor,
    "sst-2": Sst2Processor,
    "sts-b": StsbProcessor,
    "qqp": QqpProcessor,
    "qnli": QnliProcessor,
    "rte": RteProcessor,
    "wnli": WnliProcessor,
    "yelp": YelpProcessor,
    "amazon": AmazonProcessor,
    "mr": MRProcessor,
    "twitter": TwitterProcessor,
    "absa": ABSAProcessor,
    "absa_term": ABSATermProcessor
}

output_modes = {
    "cola": "classification",
    "mnli": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
    "yelp": "classification",
    "amazon": "classification",
    "mr": "classification",
    "twitter": "classification",
    "absa": "classification",
    "absa_term": "classification"
}
