# coding=utf-8
from __future__ import print_function
import argparse
import itertools
import torch
import time
import sys
import pickle
import matplotlib.pyplot as plt

from collections import OrderedDict
from torch.autograd import Variable
# import visdom
from mask_utils.global_utils import load_bert_sentences
from mask_utils.ner.loader import *
from mask_utils.ner.utils import *
from mask_utils.ner.model import BiLSTM_CRF
t = time.time()
models_path = "mask_utils/ner/models/"

parser = argparse.ArgumentParser()
parser.add_argument(
    "-T", "--train", default="data/CoNll/train.txt",
    help="Train set location"
)
parser.add_argument(
    "-d", "--dev", default="data/CoNll/valid.txt",
    help="Dev set location"
)
parser.add_argument(
    "-t", "--test", default="data/CoNll/test.txt",
    help="Test set location"
)
parser.add_argument(
    '--score', default='mask_utils/ner/evaluation/temp/score.txt',
    help='score file location'
)
parser.add_argument(
    "-s", "--tag_scheme", default="iobes",
    help="Tagging scheme (IOB or IOBES)"
)
parser.add_argument(
    "-l", "--lower", action='store_true',
    help="Lowercase words (this will not affect character inputs)"
)
parser.add_argument(
    "-z", "--zeros", action='store_true',
    help="Replace digits with 0"
)
parser.add_argument(
    "-c", "--char_dim", type=int, default=25,
    help="Char embedding dimension"
)
parser.add_argument(
    "-C", "--char_lstm_dim", type=int, default=25,
    help="Char LSTM hidden layer size"
)
parser.add_argument(
    "-b", "--char_bidirect", action='store_true',
    help="Use a bidirectional LSTM for chars"
)
parser.add_argument(
    "-w", "--word_dim", type=int, default=100,
    help="Token embedding dimension"
)
parser.add_argument(
    "-W", "--word_lstm_dim", type=int, default=100,
    help="Token LSTM hidden layer size"
)
parser.add_argument(
    "-B", "--word_bidirect", action='store_true',
    help="Use a bidirectional LSTM for words"
)
parser.add_argument(
    "-p", "--pre_emb", default="data/pre_embeddings/glove.6B.100d.txt",
    help="Location of pretrained embeddings"
)
# parser.add_argument(
    # "-A", "--all_emb", default="0",
    # type='int', help="Load all embeddings"
# )
parser.add_argument(
    "-a", "--cap_dim", type=int, default=0,
    help="Capitalization feature dimension (0 to disable)"
)
parser.add_argument(
    "-f", "--crf", action='store_true',
    help="Use CRF (0 to disable)"
)
parser.add_argument(
    "-D", "--dropout", type=float, default=0.5,
    help="Droupout on the input (0 = no dropout)"
)
parser.add_argument(
    "-r", "--reload", action='store_true',
    help="Reload the last saved model"
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
    '--name', default='test_bert_1g.model',
    help='model name'
)
parser.add_argument(
    '--char_mode', choices=['CNN', 'LSTM'], default='CNN',
    help='char_CNN or char_LSTM'
)
parser.add_argument(
    '--bert_data_dir', default='data/small_wiki_1g/final_text_files_sharded/'
)
# parser.add_argument(
    # '--mapping_file', default='models/mapping.pkl'
# )
args = parser.parse_args()

use_gpu = args.use_gpu == 1 and torch.cuda.is_available()

mapping_file = 'mask_utils/ner/models/mapping.pkl'

name = args.name
model_name = models_path + name #get_name(parameters)
tmp_model = model_name + '.tmp'

assert os.path.isfile(args.train)
assert os.path.isfile(args.dev)
assert os.path.isfile(args.test)
assert args.char_dim > 0 or args.word_dim > 0
assert 0. <= args.dropout < 1.0
assert args.tag_scheme in ['iob', 'iobes']
# assert not parameters['all_emb'] or parameters['pre_emb']
# assert not parameters['pre_emb'] or parameters['word_dim'] > 0
# assert not parameters['pre_emb'] or os.path.isfile(parameters['pre_emb'])

if not os.path.isfile(eval_script):
    raise Exception('CoNLL evaluation script not found at "%s"' % eval_script)
if not os.path.exists(eval_temp):
    os.makedirs(eval_temp)
if not os.path.exists(models_path):
    os.makedirs(models_path)

# Data parameters
lower = args.lower
zeros = args.zeros
tag_scheme = args.tag_scheme

# Load sentences
train_sentences = load_sentences(args.train, lower, zeros)
dev_sentences = load_sentences(args.dev, lower, zeros)
test_sentences = load_sentences(args.test, lower, zeros)
print("loading bert sentences")
bert_sentences = load_bert_sentences(args.bert_data_dir, zeros)

# Extract raw sentences
raw_sentences = [[x[0] for x in s] for s in (train_sentences + dev_sentences + test_sentences)]
raw_sentences.extend(bert_sentences)

# Use selected tagging scheme (IOB / IOBES)
update_tag_scheme(train_sentences, tag_scheme)
update_tag_scheme(dev_sentences, tag_scheme)
update_tag_scheme(test_sentences, tag_scheme)

# if parameters['pre_emb']:
# assume there must be pre-embeddings
dico_words = raw_word_mapping(raw_sentences, lower)
pretrained = set([line.rstrip().split()[0].strip() for line in codecs.open(args.pre_emb, 'r', 'utf-8')])
for word in pretrained:
    if word not in dico_words:
        dico_words[word] = 0
word_to_id, id_to_word = create_mapping(dico_words)


# dico_words, word_to_id, id_to_word = augment_with_pretrained(
    # dico_words_train.copy(),
    # parameters['pre_emb'],
    # list(itertools.chain.from_iterable(
        # [[w[0] for w in s] for s in dev_sentences + test_sentences])
    # ) if not parameters['all_emb'] else None
# )
# else:
    # dico_words, word_to_id, id_to_word = word_mapping(train_sentences, lower)
    # dico_words_train = dico_words

# Create a dictionary and a mapping for words / POS tags / tags
dico_chars, char_to_id, id_to_char = char_mapping(train_sentences)
dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences)

# Index data
train_data, _ = prepare_dataset(
    train_sentences, word_to_id, char_to_id, tag_to_id, lower
)
dev_data, _ = prepare_dataset(
    dev_sentences, word_to_id, char_to_id, tag_to_id, lower
)
test_data, _ = prepare_dataset(
    test_sentences, word_to_id, char_to_id, tag_to_id, lower
)

print("%i / %i / %i sentences in train / dev / test." % (
    len(train_data), len(dev_data), len(test_data)))

all_word_embeds = {}
# if parameters["pre_emb"]:
for i, line in enumerate(codecs.open(args.pre_emb, 'r', 'utf-8')):
    s = line.strip().split()
    if len(s) == args.word_dim + 1:
        all_word_embeds[s[0]] = np.array([float(i) for i in s[1:]])
word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(word_to_id), args.word_dim))

for w in word_to_id:
    if w in all_word_embeds:
        word_embeds[word_to_id[w]] = all_word_embeds[w]
    elif w.lower() in all_word_embeds:
        word_embeds[word_to_id[w]] = all_word_embeds[w.lower()]

print('Loaded %i pretrained embeddings.' % len(all_word_embeds))

with open(mapping_file, 'wb') as f:
    mappings = {
        'word_to_id': word_to_id,
        'tag_to_id': tag_to_id,
        'char_to_id': char_to_id,
        'args': args,
        'word_embeds': word_embeds,
        'embedding_dim': args.word_dim,
        'hidden_dim': args.word_lstm_dim,
        'use_crf': args.crf,
        'char_mode': args.char_mode
    }
    pickle.dump(mappings, f)
print("mapping file generate succeed")
print('word_to_id: ', len(word_to_id))
model = BiLSTM_CRF(vocab_size=len(word_to_id),
                   tag_to_ix=tag_to_id,
                   embedding_dim=args.word_dim,
                   hidden_dim=args.word_lstm_dim,
                   use_gpu=use_gpu,
                   char_to_ix=char_to_id,
                   pre_word_embeds=word_embeds,
                   use_crf=args.crf,
                   char_mode=args.char_mode)
                   # n_cap=4,
                   # cap_embedding_dim=10)
if args.reload:
    model.load_state_dict(torch.load(model_name))
if use_gpu:
    model.cuda()
learning_rate = 0.015
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
losses = []
loss = 0.0
best_dev_F = -1.0
best_test_F = -1.0
best_train_F = -1.0
all_F = [[0, 0, 0]]
plot_every = 10
eval_every = 200
count = 0
# vis = visdom.Visdom()
sys.stdout.flush()


def evaluating(model, datas, best_F):
    prediction = []
    save = False
    new_F = 0.0
    confusion_matrix = torch.zeros((len(tag_to_id) - 2, len(tag_to_id) - 2))
    for data in datas:
        ground_truth_id = data['tags']
        words = data['str_words']
        chars2 = data['chars']
        caps = data['caps']

        if args.char_mode == 'LSTM':
            chars2_sorted = sorted(chars2, key=lambda p: len(p), reverse=True)
            d = {}
            for i, ci in enumerate(chars2):
                for j, cj in enumerate(chars2_sorted):
                    if ci == cj and not j in d and not i in d.values():
                        d[j] = i
                        continue
            chars2_length = [len(c) for c in chars2_sorted]
            char_maxl = max(chars2_length)
            chars2_mask = np.zeros((len(chars2_sorted), char_maxl), dtype='int')
            for i, c in enumerate(chars2_sorted):
                chars2_mask[i, :chars2_length[i]] = c
            chars2_mask = Variable(torch.LongTensor(chars2_mask))

        if args.char_mode == 'CNN':
            d = {}
            chars2_length = [len(c) for c in chars2]
            char_maxl = max(chars2_length)
            chars2_mask = np.zeros((len(chars2_length), char_maxl), dtype='int')
            for i, c in enumerate(chars2):
                chars2_mask[i, :chars2_length[i]] = c
            chars2_mask = Variable(torch.LongTensor(chars2_mask))

        dwords = torch.LongTensor(data['words'])
        dcaps = torch.LongTensor(caps)
        if use_gpu:
            val, out = model(dwords.cuda(), chars2_mask.cuda(), dcaps.cuda(), chars2_length, d)
        else:
            val, out = model(dwords, chars2_mask, dcaps, chars2_length, d)
        predicted_id = out
        for (word, true_id, pred_id) in zip(words, ground_truth_id, predicted_id):
            line = ' '.join([word, id_to_tag[true_id], id_to_tag[pred_id]])
            prediction.append(line)
            confusion_matrix[true_id, pred_id] += 1
        prediction.append('')
    predf = eval_temp + '/pred.' + name
    scoref = eval_temp + '/score.' + name

    with open(predf, 'w') as f:
        f.write('\n'.join(prediction))

    os.system('%s < %s > %s' % (eval_script, predf, scoref))

    eval_lines = [l.rstrip() for l in codecs.open(scoref, 'r', 'utf8')]

    for i, line in enumerate(eval_lines):
        print(line)
        if i == 1:
            new_F = float(line.strip().split()[-1])
            if new_F > best_F:
                best_F = new_F
                save = True
                print('the best F is ', new_F)

    # print(("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * confusion_matrix.size(0))).format(
        # "ID", "NE", "Total",
        # *([id_to_tag[i] for i in range(confusion_matrix.size(0))] + ["Percent"])
    # ))
    # for i in range(confusion_matrix.size(0)):
        # print(("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * confusion_matrix.size(0))).format(
            # str(i), id_to_tag[i], str(confusion_matrix[i].sum()),
            # *([confusion_matrix[i][j] for j in range(confusion_matrix.size(0))] +
            #   ["%.3f" % (confusion_matrix[i][i] * 100. / max(1, confusion_matrix[i].sum()))])
        # ))
    return best_F, new_F, save

model.train(True)
for epoch in range(1, 10001):
    print("*************** epoch: {} ******************".format(epoch))
    for i, index in enumerate(np.random.permutation(len(train_data))):
        tr = time.time()
        count += 1
        data = train_data[index]
        model.zero_grad()

        sentence_in = data['words']
        sentence_in = torch.LongTensor(sentence_in)
        tags = data['tags']
        chars2 = data['chars']

        ######### char lstm
        if args.char_mode == 'LSTM':
            chars2_sorted = sorted(chars2, key=lambda p: len(p), reverse=True)
            d = {}
            for i, ci in enumerate(chars2):
                for j, cj in enumerate(chars2_sorted):
                    if ci == cj and not j in d and not i in d.values():
                        d[j] = i
                        continue
            chars2_length = [len(c) for c in chars2_sorted]
            char_maxl = max(chars2_length)
            chars2_mask = np.zeros((len(chars2_sorted), char_maxl), dtype='int')
            for i, c in enumerate(chars2_sorted):
                chars2_mask[i, :chars2_length[i]] = c
            chars2_mask = Variable(torch.LongTensor(chars2_mask))

        # ######## char cnn
        if args.char_mode == 'CNN':
            d = {}
            chars2_length = [len(c) for c in chars2]
            char_maxl = max(chars2_length)
            chars2_mask = np.zeros((len(chars2_length), char_maxl), dtype='int')
            for i, c in enumerate(chars2):
                chars2_mask[i, :chars2_length[i]] = c
            chars2_mask = torch.LongTensor(chars2_mask)


        targets = torch.LongTensor(tags)
        caps = torch.LongTensor(data['caps'])
        if use_gpu:
            neg_log_likelihood = model.neg_log_likelihood(sentence_in.cuda(), targets.cuda(), chars2_mask.cuda(), caps.cuda(), chars2_length, d)
        else:
            neg_log_likelihood = model.neg_log_likelihood(sentence_in, targets, chars2_mask, caps, chars2_length, d)
        loss += neg_log_likelihood.data.item() / len(data['words'])
        neg_log_likelihood.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
        optimizer.step()

        # if count % plot_every == 0:
            # loss /= plot_every
            # print(count, ': ', loss)
            # if losses == []:
                # losses.append(loss)
            # losses.append(loss)
            # text = '<p>' + '</p><p>'.join([str(l) for l in losses[-9:]]) + '</p>'
            # losswin = 'loss_' + name
            # textwin = 'loss_text_' + name
            # vis.line(np.array(losses), X=np.array([plot_every*i for i in range(len(losses))]),
                #  win=losswin, opts={'title': losswin, 'legend': ['loss']})
            # vis.text(text, win=textwin, opts={'title': textwin})
            # loss = 0.0

        if count % (eval_every) == 0 and count > (eval_every * 20) or \
                count % (eval_every*4) == 0 and count < (eval_every * 20):
            model.train(False)
            print(count, ": ", loss / eval_every)
            loss = 0.0
            # best_train_F, new_train_F, _ = evaluating(model, test_train_data, best_train_F)
            best_dev_F, new_dev_F, save = evaluating(model, dev_data, best_dev_F)
            if save:
                torch.save(model.state_dict(), model_name)
            # best_test_F, new_test_F, _ = evaluating(model, test_data, best_test_F)
            sys.stdout.flush()

            # all_F.append([new_train_F, new_dev_F, new_test_F])
            Fwin = 'F-score of {train, dev, test}_' + name
            # vis.line(np.array(all_F), win=Fwin,
                #  X=np.array([eval_every*i for i in range(len(all_F))]),
                #  opts={'title': Fwin, 'legend': ['train', 'dev', 'test']})
            model.train(True)

        if count % len(train_data) == 0:
            adjust_learning_rate(optimizer, lr=learning_rate/(1+0.05*count/len(train_data)))


print(time.time() - t)

# plt.plot(losses)
# plt.show()
