import numpy as np
from tqdm import tqdm

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def sigmoid_prime(X):
    return sigmoid(X) * (1 - sigmoid(X))

def softmax(X):
    Z = np.sum(np.exp(X), axis=1)
    Z = Z.reshape(Z.shape[0], 1)
    return np.exp(X) / Z

class MLP:
    def __init__(self, layer_dimensions, batch_size=100):
        self.W, self.S, self.Z, self.D, self.F = [[] for i in range(5)]
        self.num_layers = len(layer_dimensions)
        self.batch_size = batch_size
        self.W.append(np.random.normal(size=(layer_dimensions[0]+1,
                                       layer_dimensions[1]), scale=1e-4))
        self.S.append(None)
        self.Z.append(np.zeros((batch_size, layer_dimensions[0]+1)))
        self.D.append(None)
        self.F.append(None)
        for i in range(1, self.num_layers-1):
            self.W.append(np.random.normal(size=(layer_dimensions[i]+1,
                                           layer_dimensions[i+1]), scale=1e-4))
            self.S.append(np.zeros((batch_size, layer_dimensions[i]+1)))
            self.Z.append(np.zeros((batch_size, layer_dimensions[i]+1)))
            self.D.append(np.zeros((batch_size, layer_dimensions[i]+1)))
            self.F.append(np.zeros((layer_dimensions[i]+1, batch_size)))
        self.W.append(None)
        self.S.append(np.zeros((batch_size, layer_dimensions[-1])))
        self.Z.append(np.zeros((batch_size, layer_dimensions[-1])))
        self.D.append(np.zeros((batch_size, layer_dimensions[-1])))
        self.F.append(None)

    def feedforward(self, data):
        self.Z[0] = np.append(data, np.ones((data.shape[0], 1)), axis=1)
        for i in range(self.num_layers-1):
            if i == 0:
                self.S[i+1] = self.Z[i].dot(self.W[i])
            else:
                self.Z[i] = sigmoid(self.S[i])
                self.Z[i] = np.append(self.Z[i],
                                      np.ones((self.Z[i].shape[0], 1)), axis=1)
                self.F[i] = sigmoid_prime(self.S[i]).T
                self.S[i+1] = self.Z[i].dot(self.W[i])
        self.Z[-1] = softmax(self.S[-1])
        return self.Z[-1]

    def backpropagate(self, y_hat, y):
        self.D[-1] = (y_hat - y).T
        for i in range(self.num_layers-2, 0, -1):
            W_unbias = self.W[i][0:-1, :]
            self.D[i] = W_unbias.dot(self.D[i+1]) * self.F[i]
    
    def update_weights(self, eta):
        for i in range(self.num_layers-1):
            update = -eta * (self.D[i+1].dot(self.Z[i])).T
            self.W[i] += update

    def train(self, train_data, train_labels, val_data=None, val_labels=None, epochs=50, learning_rate = 0.05, model_path=None):
        save_model = True if model_path is not None else False
        num_train = len(train_labels) * len(train_labels[0])
        num_val = 0
        if val_data is not None and val_labels is not None:
            num_val = len(val_labels) * len(val_labels[0])
        print('starting training')
        for i in range(epochs):
            with tqdm(desc='epoch: {0}/{1}'.format(i+1, epochs), unit=' batches', total=len(train_data)) as pbar:
                for batch_data, batch_labels in zip(train_data, train_labels):
                    output = self.feedforward(batch_data)
                    self.backpropagate(output, batch_labels)
                    self.update_weights(eta=learning_rate)
                    pbar.update(1)

            errors = 0
            for batch_data, batch_labels in zip(train_data, train_labels):
                output = self.feedforward(batch_data)
                y_hat = np.argmax(output, axis=1)
                errors += np.sum(1-batch_labels[np.arange(len(batch_labels)), y_hat])
            out_str = 'training error: {0:.5f}'.format(float(errors)/num_train)

            if num_val > 0:
                errors = 0
                for batch_data, batch_labels in zip(val_data, val_labels):
                    output = self.feedforward(batch_data)
                    y_hat = np.argmax(output, axis=1)
                    errors += np.sum(1-batch_labels[np.arange(len(batch_labels)), y_hat])
                out_str += ', validation error: {0:.5f}'.format(float(errors)/num_val)
            
            print(out_str)
