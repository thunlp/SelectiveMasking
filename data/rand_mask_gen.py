import logging
import torch
import torch.nn as nn
import numpy as np
import spacy
import sys
import collections
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
from torch.nn.functional import softmax

sys.path.append("../")
from model.tokenization import BertTokenizer

logger = logging.getLogger(__name__)
MaskedTokenInstance = collections.namedtuple("MaskedTokenInstance", ["tokens", "info"])
MaskedItemInfo = collections.namedtuple("MaskedItemInfo", ["current_pos", "sen_doc_pos", "sen_right_id", "doc_ground_truth"])


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


class RandMask(nn.Module):
    def __init__(self, mask_rate, bert_model, do_lower_case, max_seq_length):
        super(RandMask, self).__init__()
        self.mask_rate = mask_rate
        self.max_seq_length = max_seq_length
        self.tokenizer = BertTokenizer.from_pretrained(
            bert_model, do_lower_case=do_lower_case)
        self.vocab = list(self.tokenizer.vocab.keys())

    def forward(self, data, all_labels, dupe_factor, rng):
        # data: not tokenized
        all_documents = []
        for _ in range(dupe_factor):
            for line in tqdm(data):
                all_documents.append([])
                tokens = self.tokenizer.tokenize(line)
                cand_indexes = [i for i in range(len(tokens))]
                rng.shuffle(cand_indexes)
                masked_info = [{} for token in tokens]
                masked_token = None
                masked_lms_len = 0
                num_to_predict = max(1, int(round(len(tokens) * self.mask_rate)))
                for index in cand_indexes:
                    if masked_lms_len >= num_to_predict:
                        break
                    if rng.random() < 0.8:
                        masked_token = "[MASK]"
                    else:
                        if rng.random() < 0.5:
                            masked_token = tokens[index]
                        else:
                            masked_token = self.vocab[rng.randint(0, len(self.vocab) - 1)]
                    
                    masked_info[index]["mask"] = masked_token
                    masked_info[index]["label"] = tokens[index]
                    masked_lms_len += 1
                all_documents[-1].append(MaskedTokenInstance(tokens=tokens, info=masked_info))
        return all_documents
