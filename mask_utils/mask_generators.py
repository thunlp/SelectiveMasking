import torch
import sys
import pickle
import numpy as np
import random

from .ner import loader
from .ner.model import BiLSTM_CRF
from .global_utils import mask_sentence, cap_feature, mask_1word

# from ner_model import NerModel
from torch import nn

class MaskGenerator(nn.Module):
    def __init__(self, mask_samp, mask_num, mask_rate, gpu):
        super(MaskGenerator, self).__init__()
        self.mask_samp = mask_samp
        self.mask_num = mask_num
        self.mask_rate = mask_rate
        self.gpu = gpu

    def evaluate(self, data):
        pass

    def generate_mask(self, data):
        pass

    def forward(self):
        pass


class Ner(MaskGenerator):
    def __init__(self, config):
        super(Ner, self).__init__(config["mask_samp"], config["mask_num"], config["mask_rate"], config["gpu"])
        mappings = self.load_mapping(config["mapping_file"])
        self.word_to_id = mappings['word_to_id']
        self.D = {}
        for key, value in self.word_to_id.items():
            self.D[value] = key
        self.tag_to_id = mappings['tag_to_id']
        self.id_to_tag = {k[1]: k[0] for k in self.tag_to_id.items()}
        self.char_to_id = mappings['char_to_id']
        self.word_embeds = mappings['word_embeds']
        train_config = mappings['args']
        self.lower = train_config.lower
        self.zeros = train_config.zeros
        self.tag_scheme = train_config.tag_scheme
        self.embedding_dim = train_config.word_dim
        self.hidden_dim = train_config.word_lstm_dim
        self.use_crf = train_config.crf
        self.char_mode = train_config.char_mode
        self.vocab_size = len(self.word_to_id)

        self.model = BiLSTM_CRF(vocab_size=self.vocab_size, 
                                tag_to_ix=self.tag_to_id,
                                embedding_dim=self.embedding_dim,
                                hidden_dim=self.hidden_dim,
                                use_gpu=self.gpu,
                                char_to_ix=self.char_to_id,
                                pre_word_embeds=self.word_embeds,
                                use_crf=self.use_crf,
                                char_mode=self.char_mode)
        # print(self.gpu)
        if self.gpu:
            self.model.load_state_dict(torch.load(config["model_path"], map_location="cuda"))
            self.model.cuda()
        else:
            self.model.load_state_dict(torch.load(config["model_path"], map_location="cpu"))

        self.model.eval()

    def load_mapping(self, mapping_file):
        with open(mapping_file, 'rb') as f:
            mappings = pickle.load(f)
        return mappings

    def prepare_data(self, tokens):
        def f(x): return x.lower() if self.lower else x
        words = [self.word_to_id[f(w) if f(w) in self.word_to_id else '<UNK>']
                 for w in tokens]
        # Skip characters that are not in the training set
        chars = [[self.char_to_id[c] for c in w if c in self.char_to_id] for w in tokens]
        caps = [cap_feature(w) for w in tokens]
        data = {
            'str_words': tokens,
            'words': words,
            'chars': chars,
            'caps': caps,
        }

        # masked data
        masked_datas = []
        # for samp in range(self.mask_samp):
        for pos in range(len(tokens)):
            masked_data, masked_poses = [], []
            new_s, mask_pos = mask_1word(tokens, pos)
            # new_s, mask_pos = mask_sentence(tokens, self.mask_num, self.mask_rate)
            words = [self.word_to_id[f(w) if f(w) in self.word_to_id else '<UNK>']
                     for w in new_s]
            # Skip characters that are not in the training set
            chars = [[self.char_to_id[c] for c in w if c in self.char_to_id] for w in new_s]
            caps = [cap_feature(w) for w in new_s]
            masked_datas.append({
                'data': {
                    'str_words': new_s,
                    'words': words,
                    'chars': chars,
                    'caps': caps,
                },
                'pos': mask_pos
            })

        return data, masked_datas

    def evaluate(self, data):
        # print(data)
        words = data['str_words']
        chars2 = data['chars']
        caps = data['caps']
        # assume char mode is CNN
        d = {}
        chars2_length = [len(c) for c in chars2]
        char_maxl = max(chars2_length)
        chars2_mask = np.zeros((len(chars2_length), char_maxl), dtype='int')
        for i, c in enumerate(chars2):
            chars2_mask[i, :chars2_length[i]] = c
        chars2_mask = torch.LongTensor(chars2_mask)

        dwords = torch.LongTensor(data['words'])
        dcaps = torch.LongTensor(caps)
        print("OK")
        if self.gpu:
            val, out = self.model(dwords.cuda(), chars2_mask.cuda(), dcaps.cuda(), chars2_length, d)
        else:
            val, out = self.model(dwords, chars2_mask, dcaps, chars2_length, d)
        print(val)
        print(out)
        exit(0)
        pred_result = [(word, self.id_to_tag[pred_id]) for (word, pred_id) in zip(words, out)]
        
        return pred_result

    def evaluate_batch(self, datas):
        words_b = [data["data"]['str_words'] for data in datas]
        chars2_b = [data["data"]['chars'] for data in datas]
        caps_b = [data["data"]['caps'] for data in datas]

        d = {}
        chars2_length_b = [[len(c) for c in chars2] for chars2 in chars2_b]
        char_maxl_b = [max(chars2_length) for chars2_length in chars2_length_b]
        chars2_mask_b = [np.zeros((len(chars2_length), char_maxl), dtype='int') for (chars2_length, char_maxl) in zip(chars2_length_b, char_maxl_b)]
        for (batch_index, chars2) in enumerate(chars2_b):
            for i, c in enumerate(chars2):
                chars2_mask_b[batch_index][i, :chars2_length_b[batch_index][i]] = c
        chars2_mask_b = torch.LongTensor(chars2_mask_b)

        dwords_b = torch.LongTensor([data["data"]["words"] for data in datas])
        dcaps_b = torch.LongTensor(caps_b)
        
        if self.gpu:
            val, out_b = self.model(dwords_b.cuda(), chars2_mask_b.cuda(), dcaps_b.cuda(), chars2_length_b, d)
        else:
            val, out_b = self.model(dwords_b, chars2_mask_b, dcaps_b, chars2_length_b, d)

        pred_results = [[(word, self.id_to_tag[pred_id]) for (word, pred_id) in zip(words, out)] for (words, out) in zip(words_b, out_b)]

        return pred_results

    def generate_mask(self, data, masked_datas):
        try:
            prediction = self.evaluate_batch([{"data": data}])
        except ValueError:
            print("OOOO")
        pos_signi = [0 for w in data['words']]

        try:
            masked_predictions = self.evaluate_batch(masked_datas)
        except ValueError:
            print(masked_datas)
            raise ValueError

        for i in range(len(masked_datas)):
            masked_prediction = masked_predictions[i]
            masked_data = masked_datas[i]
            diff_words_num = 0
            index_masked = 0
            for index in range(len(prediction)):
                if index not in masked_data["pos"]:
                    assert prediction[index][0] == masked_prediction[index_masked][0]
                    if prediction[index][1] != masked_prediction[index_masked][1]:
                        diff_words_num += 1
                    index_masked += 1

            for pos in masked_data["pos"]:
                pos_signi[pos] += diff_words_num

        # for masked_data in masked_datas:
        #     # print(masked_data['pos'])
        #     try:
        #         masked_prediction = self.evaluate(masked_data['data'])
        #     except ValueError:
        #         print(masked_datas)
        #         raise ValueError
        #     diff_words_num = 0
        #     # print(prediction)
        #     # print(masked_prediction)
        #     index_masked = 0
        #     for index in range(len(prediction)):
        #         if index not in masked_data['pos']:
        #             assert prediction[index][0] == masked_prediction[index_masked][0]
        #             if prediction[index][1] != masked_prediction[index_masked][1]:
        #                 diff_words_num += 1
        #             index_masked += 1
            # if diff_words_num > 0:
                # print(self.D[data['words'][masked_data['pos'][0]]])
                # print([item[1] for item in prediction])
                # print([item[1] for item in masked_prediction])
        #    for pos in masked_data["pos"]:
        #        pos_signi[pos] += diff_words_num
        # print([self.D[w] for w in data["words"]])
        # print(pos_signi)
        L = []
        for i in range(len(pos_signi)):
            if pos_signi[i] > 0:
                L.append(self.D[data["words"][i]])
        # print(L)

        zip_list = list(enumerate(pos_signi))
        random.shuffle(zip_list)
        signi_indexes = tuple(zip(*sorted(zip_list, key=lambda x: x[1], reverse=True)))[0]
        return signi_indexes

    def forward(self, line):
        data, masked_datas = self.prepare_data(line.strip().split())
        # print(data, masked_datas)
        signi_indexes = self.generate_mask(data, masked_datas)
        return signi_indexes
