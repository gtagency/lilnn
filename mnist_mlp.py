import pickle, gzip
import numpy as np
from lilnn import MLP

def label_to_bit_vector(labels, nbits):
    bv = np.zeros((labels.shape[0], nbits))
    for i in range(labels.shape[0]):
        bv[i, labels[i]] = 1.0

    return bv

def create_minibatches(data, labels, batch_size, create_bit_vector=False):
    N = data.shape[0]
    if N % batch_size != 0:
        print( "Warning in create_minibatches(): Batch size {0} does not " \
              "evenly divide the number of examples {1}.".format(batch_size,
                                                                 N))
    chunked_data = []
    chunked_labels = []
    idx = 0
    while idx + batch_size <= N:
        chunked_data.append(data[idx:idx+batch_size, :])
        if not create_bit_vector:
            chunked_labels.append(labels[idx:idx+batch_size])
        else:
            bv = label_to_bit_vector(labels[idx:idx+batch_size], 10)
            chunked_labels.append(bv)

        idx += batch_size

    return chunked_data, chunked_labels

print("loading data...")
f = gzip.open('mnist.pkl.gz')
train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
f.close()

minibatch_size = 100
train_data, train_labels = create_minibatches(train_set[0], train_set[1],
                                              minibatch_size,
                                              create_bit_vector=True)
valid_data, valid_labels = create_minibatches(valid_set[0], valid_set[1],
                                              minibatch_size,
                                              create_bit_vector=True)

mlp = MLP(layer_dimensions=[784, 100, 100, 10], batch_size=minibatch_size)
mlp.train(train_data, train_labels, valid_data, valid_labels)
