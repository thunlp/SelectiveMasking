import logging 
import torch
import torch.nn as nn
import numpy as np
import spacy
import collections
from spacy.lang.en import English
from tqdm import tqdm
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler

from modeling import BertForSequenceClassification
from tokenization import BertTokenizer

logger = logging.getLogger(__name__)
MaskedTokenInstance = collections.namedtuple("MaskedTokenInstance", ["tokens", "info"])


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

class SC(nn.Module):
    def __init__(self, mask_rate, top_sen_rate, top_token_rate, bert_model, do_lower_case, max_seq_length, label_list, sen_batch_size, use_gpu=True):
        super(SC, self).__init__()
        self.mask_rate = mask_rate
        self.top_sen_rate = top_sen_rate
        self.top_token_rate = top_token_rate
        self.label_list = label_list
        self.num_labels = len(self.label_list)
        self.max_seq_length = max_seq_length
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)
        self.model = BertForSequenceClassification.from_pretrained(bert_model, num_labels=self.num_labels)
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        print(self.device)
        self.model.to(self.device)
        self.n_gpu = torch.cuda.device_count()
        self.sen_batch_size = sen_batch_size
        self.vocab = list(self.tokenizer.vocab.keys())
        if self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

    def convert_examples_to_features(self, data):
        features = []
        for (ex_index, tokens_a) in enumerate(data):
            if ex_index % 10000 == 0:
                logger.info("Writing example %d of %d" % (ex_index, len(data)))
            if len(tokens_a) > self.max_seq_length - 2:
                tokens_a = tokens_a[:(self.max_seq_length - 2)]
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            
            segment_ids = [0] * len(tokens)
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            
            padding = [0] * (self.max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == self.max_seq_length
            assert len(input_mask) == self.max_seq_length
            assert len(segment_ids) == self.max_seq_length

            features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids))

        return features

    def evaluate(self, data, batch_size):
        # print(data)
        eval_features = self.convert_examples_to_features(data)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

        self.model.eval()
        preds = []
        for input_ids, input_mask, segment_ids, in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)
            with torch.no_grad():
                logits = self.model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)

        preds_arg = np.argmax(preds[0], axis=1)
        return preds_arg, preds[0]

    def mask_token(self, sen):
        masked_sentences = []
        masked_poses = []
        for i in range(len(sen)):
            masked_poses.append(i)
            masked_sentences.append([sen[j] for j in range(len(sen)) if j != i])
        return masked_sentences, masked_poses

    def create_mask(self, mask_poses, sen, rng):
        masked_info = [{} for token in sen]
        for pos in mask_poses:
            if rng.random() < 0.8:
                mask_token = "[MASK]"
            else:
                if rng.random() < 0.5:
                    mask_token = sen[pos]
                else:
                    mask_token = self.vocab[rng.randint(0, len(self.vocab) - 1)]
            masked_info[pos]["mask"] = mask_token
            masked_info[pos]["label"] = sen[pos]
        return masked_info
        

    def forward(self, data, all_labels, rng):
        # data: not tokenized
        # convert label to ids
        doc_num = len(data)
        label_map = {label : i for i, label in enumerate(self.label_list)}
        all_label_ids = [label_map[label] for label in all_labels]
        
        # convert data, segment data to sentences
        tokenized_data = []
        nlp = English()
        sentencizer = nlp.create_pipe("sentencizer")
        nlp.add_pipe(sentencizer)
        sentences = []
        sen_doc_ids = [] # [0, 0, ..., 0, 1, 1, ..., 1, ...]
        # which_select = []
        for (doc_id, doc) in enumerate(data):
            tokenized_data.append(self.tokenizer.tokenize(doc))
            doc = nlp(doc)
            tL = [self.tokenizer.tokenize(sen.text) for sen in doc.sents]
            sentences.extend(tL)
            sen_doc_ids.extend([doc_id] * len(tL))
            # which_select.extend([False] * len(sens))
        
        # logger.info("Begin eval for all doc")
        # doc_preds, _ = self.evaluate(tokenized_data, self.doc_batch_size)
        logger.info("Begin eval for all sentence")
        sens_preds, sens_pred_scores = self.evaluate(sentences, self.sen_batch_size)
        right_sens = []
        right_preds = []
        right_scores = []
        right_sen_doc_ids = []
        right_sen_doc_poses = []
        i = 0
        for doc_id in range(doc_num):
            ds = []
            while i < len(sen_doc_ids) and sen_doc_ids[i] == doc_id:
                sen_pred = sens_preds[i]
                doc_ground_truth = all_label_ids[doc_id]
                # compare with ground truth
                if doc_ground_truth == sen_pred:
                    # (sentence, doc_id, sen_doc_pos, pred, score of ground truth)
                    ds.append((sentences[i], doc_id, i, sen_pred, sens_pred_scores[i][doc_ground_truth]))
                    # which_select[i] = True
                i += 1
            if len(ds) == 0:
                continue
            t_sen, t_sen_doc_id, t_sen_doc_pos, t_pred, t_score = zip(*ds[0:max(int(self.top_sen_rate * len(ds)), 1)])  # select top sentences
            right_sens.extend(t_sen)
            right_preds.extend(t_pred)
            right_scores.extend(t_score)
            right_sen_doc_ids.extend(t_sen_doc_id)
            right_sen_doc_poses.extend(t_sen_doc_pos)
        
        right_sens_num = len(right_sens)
        masked_sens = []
        masked_poses = []
        mask_sen_doc_poses = []
        for sen_doc_pos, sen in zip(right_sen_doc_poses, right_sens):
            masked_sen, masked_pos = self.mask_token(sen)
            masked_sens.extend(masked_sen)
            masked_poses.extend(masked_pos)
            mask_sen_doc_poses.extend([sen_doc_pos] * len(masked_sen))

        mask_sens_preds, mask_sens_scores = self.evaluate(masked_sens, self.sen_batch_size)
        i = 0
        mask_pos_d = {}
        for sen_id in range(right_sens_num):
            st = []
            origin_score = right_scores[sen_id]
            sen_doc_pos = right_sen_doc_poses[sen_id]
            doc_ground_truth = all_label_ids[sen_doc_ids[sen_doc_pos]]
            while i < len(mask_sen_doc_poses) and mask_sen_doc_poses[i] == sen_doc_pos:
                c = origin_score - mask_sens_scores[i][doc_ground_truth]
                # (masked_pos, score)
                st.append((masked_poses[i], c))
                i += 1
            st = sorted(st, key=lambda x: x[-1], reverse=True)
            mask_pos_d[sen_doc_pos], _ = zip(*st)
        
        all_documents = []
        i = 0
        for doc_id in tqdm(range(doc_num), desc="Generating All Documents"):
            all_documents.append([])
            while i < len(sen_doc_ids) and doc_id == sen_doc_ids[i]:
                m_info = []
                if i in mask_pos_d:
                    m_info = self.create_mask(mask_pos_d[i], sentences[i], rng)
                all_documents[-1].append(MaskedTokenInstance(tokens=sentences[i], info=m_info))
                i += 1

        return all_documents
