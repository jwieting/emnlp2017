import lasagne
import random
import numpy as np
import sys
import argparse
import io
import utils
from models import models
from example import example

def get_data(params):
    lines = io.open(params.data, 'r', encoding='utf-8').readlines()
    scores = {}
    random.shuffle(lines)
    for i in lines:
        i = i.lower()
        i = i.split("\t")
        score = float(i[2])
        idx = i[3]
        if idx not in scores and score <= params.max_value and score >= params.min_value:
            ex = (example(i[0]), example(i[1]))
            scores[idx] = ex
    data = []
    for i in scores:
        data.append(scores[i])
    return data

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
parser.add_argument("-LW", help="Regularization on embedding parameters", type=float, default=0.)
parser.add_argument("-outfile", help="Name of output file")
parser.add_argument("-batchsize", help="Size of batch", type=int, default=100)
parser.add_argument("-dim", help="Dimension of model", type=int, default=300)
parser.add_argument("-wordfile", help="Word embedding file", default="../data/paragram_sl999_small.txt")
parser.add_argument("-save", help="Whether to pickle model", type=int, default=0)
parser.add_argument("-margin", help="Margin in objective function", type=float, default=0.4)
parser.add_argument("-samplingtype", help="Type of Sampling used: MAX, MIX, or RAND", default="MAX")
parser.add_argument("-evaluate", help="Whether to evaluate the model during training", type=int, default=1)
parser.add_argument("-epochs", help="Number of epochs in training", type=int, default=10)
parser.add_argument("-eta", help="Learning rate", type=float, default=0.001)
parser.add_argument("-learner", help="Either AdaGrad or Adam", default="adam")
parser.add_argument("-model", help="Which model to use, either gran or wordaverage")
parser.add_argument("-gran_type", type=int,  help="Type of GRAN model", default=1)
parser.add_argument("-min_value", type=float, help="Min cut-off value used in filtering data", default = 0.)
parser.add_argument("-max_value", type=float, help="Max cut-off value used in filtering data", default = 1.)
parser.add_argument("-data", help="Name of data file containing paraphrases", default=None)

args = parser.parse_args()
args.learner = str2learner(args.learner)
print " ".join(sys.argv)

params = args

data = get_data(params)
words, We = utils.get_wordmap(args.wordfile)

model = models(We, params)

print "Num examples:", len(data)
print "Num words:", len(words)

model.train(data, words, params)