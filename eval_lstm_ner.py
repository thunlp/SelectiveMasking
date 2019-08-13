# coding=utf-8
from __future__ import print_function
import argparse
import torch
import time
import pickle
import os
import sys
from torch.autograd import Variable
from tqdm import tqdm
from mask_utils.ner.loader import *
from mask_utils.ner.utils import *
from mask_utils.ner.model import BiLSTM_CRF


t = time.time()

# python -m visdom.server

parser = argparse.ArgumentParser()
parser.add_argument(
    "-t", "--test", default="data/CoNll/test.txt",
    help="Test set location"
)
parser.add_argument(
    '--dir', default='',
    help='score file location'
)
parser.add_argument(
    "-f", "--crf", action='store_true',
    help="Use CRF (0 to disable)"
)
parser.add_argument(
    "-g", '--use_gpu', action='store_true',
    help='whether or not to ues gpu'
)
parser.add_argument(
    '--loss', default='loss.txt',
    help='loss file location'
)
parser.add_argument(
    '--char_mode', choices=['CNN', 'LSTM'], default='CNN',
    help='char_CNN or char_LSTM'
)
parser.add_argument(
    '--model_name', type=str
)
parser.add_argument(
    '--eval_script', default='mask_utils/ner/evaluation/conlleval'
)


args = parser.parse_args()

mapping_file = os.path.join(args.dir, "mapping.pkl")

with open(mapping_file, 'rb') as f:
    mappings = pickle.load(f)

word_to_id = mappings['word_to_id']
tag_to_id = mappings['tag_to_id']
id_to_tag = {k[1]: k[0] for k in tag_to_id.items()}
char_to_id = mappings['char_to_id']
config = mappings['args']
word_embeds = mappings['word_embeds']

assert os.path.isfile(args.test)
assert config.tag_scheme in ['iob', 'iobes']

if not os.path.isfile(args.eval_script):
    raise Exception('CoNLL evaluation script not found at "%s"' % eval_script)

lower = config.lower
zeros = config.zeros
tag_scheme = config.tag_scheme
# mask_rate = parameters['mask_rate']

test_sentences = load_sentences(args.test, lower, zeros)
update_tag_scheme(test_sentences, tag_scheme)
test_data = prepare_dataset(
    test_sentences, word_to_id, char_to_id, tag_to_id, lower)

model = BiLSTM_CRF(vocab_size=len(word_to_id),
                    tag_to_ix=tag_to_id,
                    embedding_dim=config.word_dim,
                    hidden_dim=config.word_lstm_dim,
                    use_gpu=args.use_gpu,
                    char_to_ix=char_to_id,
                    pre_word_embeds=word_embeds,
                    use_crf=config.crf,
                    char_mode=config.char_mode)

model_path = os.path.join(args.dir, args.model_name)

if args.use_gpu:
    model.load_state_dict(torch.load(model_path, map_location="cuda"))
    model.cuda()
else:
    model.load_state_dict(toch.load(model_path, map_location="cpu"))
    model.cpu()
model.eval() #NOTE very important

def evaluate(model, datas, postfix=''):
    prediction = []
    confusion_matrix = torch.zeros((len(tag_to_id) - 2, len(tag_to_id) - 2))
    # print("OK")
    for data in tqdm(datas):
        ground_truth_id = data['tags']
        words = data['str_words']
        chars2 = data['chars']
        caps = data['caps']

        if config.char_mode == 'LSTM':
            chars2_sorted = sorted(chars2, key=lambda p: len(p), reverse=True)
            d = {}
            for i, ci in enumerate(chars2):
                for j, cj in enumerate(chars2_sorted):
                    if ci == cj and not j in d and not i in d.values():
                        d[j] = i
                        continue
            chars2_length = [len(c) for c in chars2_sorted]
            char_maxl = max(chars2_length)
            chars2_mask = np.zeros(
                (len(chars2_sorted), char_maxl), dtype='int')
            for i, c in enumerate(chars2_sorted):
                chars2_mask[i, :chars2_length[i]] = c
            chars2_mask = Variable(torch.LongTensor(chars2_mask))

        if config.char_mode == 'CNN':
            d = {}
            chars2_length = [len(c) for c in chars2]
            char_maxl = max(chars2_length)
            chars2_mask = np.zeros(
                (len(chars2_length), char_maxl), dtype='int')
            for i, c in enumerate(chars2):
                chars2_mask[i, :chars2_length[i]] = c
            chars2_mask = Variable(torch.LongTensor(chars2_mask))

        dwords = torch.LongTensor(data['words'])
        dcaps = torch.LongTensor(caps)
        if args.use_gpu:
            val, out = model(dwords.cuda(), chars2_mask.cuda(),
                             dcaps.cuda(), chars2_length, d)
        else:
            val, out = model(dwords, chars2_mask, dcaps, chars2_length, d)
        predicted_id = out
        for (word, true_id, pred_id) in zip(words, ground_truth_id, predicted_id):
            line = ' '.join([word, id_to_tag[true_id], id_to_tag[pred_id]])
            prediction.append(line)
            confusion_matrix[true_id, pred_id] += 1
        prediction.append('')
    predf = os.path.join(args.dir, 'test_pred.' + args.model_name)
    scoref = os.path.join(args.dir, 'test_score.' + args.model_name)
    with open(predf, 'w') as f:
        f.write('\n'.join(prediction))

    os.system('%s < %s > %s' % (args.eval_script, predf, scoref))

evaluate(model, test_data)

print("Evaluation time: {}".format(time.time() - t))
