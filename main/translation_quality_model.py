import theano
import numpy as np
import time
import lasagne
import cPickle
from theano import tensor as T
from theano import config
from sklearn.metrics import accuracy_score
from lasagne_layers import lasagne_average_layer

class translation_quality_model(object):

    def prepare_data(self, list_of_seqs):
        lengths = [len(s) for s in list_of_seqs]
        n_samples = len(list_of_seqs)
        maxlen = np.max(lengths)
        x = np.zeros((n_samples, maxlen)).astype('int32')
        x_mask = np.zeros((n_samples, maxlen)).astype(theano.config.floatX)
        for idx, s in enumerate(list_of_seqs):
            x[idx, :lengths[idx]] = s
            x_mask[idx, :lengths[idx]] = 1.
        x_mask = np.asarray(x_mask, dtype=config.floatX)
        return x, x_mask

    def save_params(self, fname, words):
        f = file(fname, 'wb')
        values = lasagne.layers.get_all_param_values(self.final_layer)
        values.append(words)
        cPickle.dump(values, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()

    def get_minibatches_idx(self, n, minibatch_size, shuffle=False):
        idx_list = np.arange(n, dtype="int32")

        if shuffle:
            np.random.shuffle(idx_list)

        minibatches = []
        minibatch_start = 0
        for i in range(n // minibatch_size):
            minibatches.append(idx_list[minibatch_start:
            minibatch_start + minibatch_size])
            minibatch_start += minibatch_size

        if (minibatch_start != n):
            minibatches.append(idx_list[minibatch_start:])

        return zip(range(len(minibatches)), minibatches)

    def prepare(self, batch, params):

        g = []

        for i in batch:
            g.append((i[0].embeddings, True))
            if params.pairing_type == "random":
                neg = np.random.choice(i[1])
                g.append((neg.embeddings, False))
            else:
                g2 = []
                for j in i[1]:
                    g2.append(j.embeddings)
                x, m = self.prepare_data(g2)
                scores = self.predict_function(x,m)
                t = np.argmax(scores,axis=0)[0]
                g.append((g2[t], False))

        gx, gmask = self.prepare_data([i[0] for i in g])
        scores = np.zeros((len(g),2))
        for i in range(len(g)):
            if g[i][1]:
                scores[i][0] = 1
            else:
                scores[i][1] = 1

        scores = np.array(scores, dtype = bool)
        if gx.shape[1] > params.max_len:
            gx = gx[:,0:params.max_len]
            gmask = gmask[:,0:params.max_len]

        return (gx, gmask, scores)

    def prepare_eval(self, batch, ids, params):

        g = []
        t_idxs = []

        for n,i in enumerate(batch):
            g.append((i[0].embeddings,True))
            t = i[1][ids[n]]
            t_idxs.extend(t.embeddings)
            g.append((t.embeddings,False))

        gx, gmask = self.prepare_data([i[0] for i in g])
        scores = np.zeros((len(g),2))
        for i in range(len(g)):
            if g[i][1]:
                scores[i][0] = 1
            else:
                scores[i][1] = 1

        scores = np.array(scores, dtype = bool)

        if gx.shape[1] > params.max_len:
            gx = gx[:,0:params.max_len]
            gmask = gmask[:,0:params.max_len]

        return (gx, gmask, scores)

    def __init__(self, We_initial, params):

        We = theano.shared(np.asarray(We_initial, dtype=config.floatX))

        gidx = T.imatrix()
        gmask = T.matrix()
        scores = T.matrix()

        l_in = lasagne.layers.InputLayer((None, None))
        l_mask = lasagne.layers.InputLayer(shape=(None, None))
        l_emb = lasagne.layers.EmbeddingLayer(l_in, input_size=We.get_value().shape[0],
                                              output_size=We.get_value().shape[1], W=We)

        if params.model == "wordaverage":
            l_out = lasagne_average_layer([l_emb, l_mask], tosum=False)

        elif params.model == "lstm":
            l_lstm = lasagne.layers.LSTMLayer(l_emb, params.dim, peepholes=True, learn_init=False,
                                              mask_input=l_mask)
            l_out = lasagne.layers.SliceLayer(l_lstm, -1, 1)

        l_softmax = lasagne.layers.DenseLayer(l_out, 2, nonlinearity=T.nnet.softmax)
        X = lasagne.layers.get_output(l_softmax, {l_in:gidx, l_mask:gmask})
        cost = T.nnet.categorical_crossentropy(X,scores)

        network_params = lasagne.layers.get_all_params(l_out, trainable=True)
        network_params.pop(0)

        self.all_params = lasagne.layers.get_all_params(l_out, trainable=True)
        self.final_layer = l_softmax
        print self.all_params

        l2 = 0.5 * params.LC * sum(lasagne.regularization.l2(x) for x in network_params)
        cost = T.mean(cost) + l2

        grads = theano.gradient.grad(cost, self.all_params)
        updates = params.learner(grads, self.all_params, params.eta)

        self.train_function = theano.function([gidx, gmask, scores], cost, updates=updates)
        self.predict_function = theano.function([gidx, gmask], X)

        print "Num Params:", lasagne.layers.count_params(self.final_layer)

    def evaluate(self, val_data, test_data, idxs, words, params):

        def get_accuracy(data, idxs):
            kf = self.get_minibatches_idx(len(data), params.batchsize, shuffle = False)
            golds = []
            preds = []
            for _, train_index in kf:
                batch = [data[t] for t in train_index]

                for i in batch:
                    i[0].populate_embeddings(words, True)
                    for j in i[1]:
                        j.populate_embeddings(words, True)

                (gx, gmask, scores) = self.prepare_eval(batch, [idxs[t] for t in train_index], params)

                preds_ = self.predict_function(gx, gmask)
                golds_ = []
                for i in range(scores.shape[0]):
                    if scores[i][0]:
                        golds.append(0)
                    else:
                        golds.append(1)
                preds.extend(preds_)
                golds.extend(golds_)

            preds = np.argmax(preds,axis=1)

            one_idx = np.where(np.array(golds) == 1)[0]
            zero_idx = np.where(np.array(golds) == 0)[0]

            return accuracy_score(golds,preds), accuracy_score([golds[i] for i in zero_idx], [preds[i] for i in zero_idx]), \
                   accuracy_score([golds[i] for i in one_idx], [preds[i] for i in one_idx])

        acc_val, _, _ = get_accuracy(val_data, idxs['val'])
        acc_test, acc_0, acc_1 = get_accuracy(test_data, idxs['test'])

        print "val: {0}, test: {1} | {2} {3}".format(acc_val, acc_test, acc_0, acc_1)
        return acc_val

    def train(self, train_data, val_data, test_data, idxs, words, params):

        start_time = time.time()
        v = self.evaluate(val_data, test_data, idxs, words, params)
        max_v = v

        try:
            for eidx in xrange(params.epochs):

                kf = self.get_minibatches_idx(len(train_data), params.batchsize, True)
                uidx = 0
                for _, train_index in kf:

                    uidx += 1

                    batch = [train_data[t] for t in train_index]

                    for i in batch:
                        i[0].populate_embeddings(words, True)
                        for j in i[1]:
                            j.populate_embeddings(words, True)

                    (gx, gmask, scores) = self.prepare(batch, params)

                    cost = self.train_function(gx, gmask, scores)

                    if np.isnan(cost) or np.isinf(cost):
                        print 'NaN detected'

                    for i in batch:
                        i[0].unpopulate_embeddings()
                        for j in i[1]:
                            j.unpopulate_embeddings()

                if params.evaluate:
                    v = self.evaluate(val_data, test_data, idxs, words, params)

                if params.save and max_v > v:
                    self.save_params(params.outfile + '.pickle', words)
                    max_v = v

                print 'Epoch ', (eidx + 1), 'Cost ', cost

        except KeyboardInterrupt:
            print "Training interupted"

        end_time = time.time()
        print "total time:", (end_time - start_time)
