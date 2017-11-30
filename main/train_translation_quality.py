import utils
import lasagne
import random
import numpy as np
import sys
import argparse
import io
from translation_quality_model import translation_quality_model
from example import example

def get_data(params):
    lines = io.open(params.data, 'r', encoding='utf-8').readlines()

    d = {}
    data = []
    idx = 0

    for i in lines:
        i = i.strip()
        if len(i) == 0:
            continue
        i = i.split('\t')
        if len(i) == 1:
            d[idx] = (i[0], [])
            idx += 1
        else:
            d[idx - 1] = (d[idx - 1][0], d[idx - 1][1] + [i[0]])

    for i in d.items():
        r = i[1][0]
        trans = i[1][1]
        r = example(r)
        trans = [example(j) for j in trans]
        data.append((i[0], r, trans))

    train = []; test = []; val = []
    for i in data:
        idx, r, t = i
        if idx < 40000:
            train.append((i[1],i[2]))
        elif idx < 45000:
            val.append((i[1],i[2]))
        elif idx < 50000:
            test.append((i[1],i[2]))

    return train, val, test

def get_idxs(data):
    lis = []
    for i in data:
        r = random.randint(0,len(i[1])-1)
        lis.append(r)
    return lis

def str2learner(v):
    if v is None:
        return None
    if v.lower() == "adagrad":
        return lasagne.updates.adagrad
    if v.lower() == "adam":
        return lasagne.updates.adam
    raise ValueError('A type that was supposed to be a learner is not.')

random.seed(1)
np.random.seed(1)

parser = argparse.ArgumentParser()

parser.add_argument("-LC", help="Regularization on composition parameters", type=float, default=0.)
parser.add_argument("-outfile", help="Name of output file")
parser.add_argument("-batchsize", help="Size of batch", type=int, default=100)
parser.add_argument("-dim", help="Dimension of model", type=int, default=300)
parser.add_argument("-wordfile", help="Word embedding file", default="../data/paragram_sl999_small.txt")
parser.add_argument("-save", help="Whether to pickle model", type=int, default=0)
parser.add_argument("-evaluate", help="Whether to evaluate the model during training", type=int, default=1)
parser.add_argument("-epochs", help="Number of epochs in training", type=int, default=10)
parser.add_argument("-eta", help="Learning rate", type=float, default=0.001)
parser.add_argument("-learner", help="Either AdaGrad or Adam", default="adam")
parser.add_argument("-model", help="Which model to use either lstm or wordaverage")
parser.add_argument("-data", help="Name of data file containing paraphrases", default=None)
parser.add_argument("-pairing_type", help="Type of selection of negative examples, either random or max", default="random")
parser.add_argument("-max_len", help="Max length of sentences, longer sentences are trimmed to this length", default="random")

args = parser.parse_args()
args.learner = str2learner(args.learner)
print " ".join(sys.argv)

params = args

train_data, val_data, test_data = get_data(params)

idxs = {}
idxs['val'] = get_idxs(val_data)
idxs['test'] = get_idxs(test_data)

(words, We) = utils.get_wordmap(params.wordfile)

model = translation_quality_model(We, params)

print "Num examples:", len(train_data)

model.train(train_data, val_data, test_data, idxs, words, params)