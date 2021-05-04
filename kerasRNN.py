import numpy as np
from keras.preprocessing.text import Tokenizer
from preprocess import cleaner
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Dense, Input, LSTM, Bidirectional, Activation, Conv1D, GRU, TimeDistributed
from keras.layers import Dropout, Embedding, GlobalMaxPooling1D, MaxPooling1D, Add, Flatten, SpatialDropout1D
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, SimpleRNN, Activation
from keras.models import Sequential
from keras import optimizers
from sklearn.metrics import accuracy_score


class KerasRNNClassifier:

    def __init__(self, max_words=30000, input_length=20, emb_dim=300, n_classes=3, epochs=15, batch_size=64, emb_idx=0):
        self.max_words = max_words
        self.input_length = input_length
        self.emb_dim = emb_dim
        self.n_classes = n_classes
        self.epochs = epochs
        self.bs = batch_size
        self.embeddings_index = emb_idx
        self.tokenizer = Tokenizer(num_words=self.max_words + 1, lower=True, split=' ')
        self.model = self._rnn()

    def _preprocess(self, texts):
        return [cleaner(x) for x in texts]

    def _rnn(self):
        model = Sequential()
        model.add(SimpleRNN(50, input_shape=(49, 1), return_sequences=False))
        model.add(Dense(46))
        model.add(Activation('softmax'))

        adam = optimizers.Adam(lr=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

        return model

    def _get_sequences(self, texts):
        seqs = self.tokenizer.texts_to_sequences(texts)
        seqs = pad_sequences(seqs, maxlen=self.input_length, value=0)
        return seqs

    def _predict_probability(self, X, y=None):
        seqs = self._get_sequences(self._preprocess(X))
        return self.model.predict(seqs)

    def _predict(self, X, y=None):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(np.argmax(y, axis=1), y_pred)

    def fit(self, X, y):
        y = to_categorical(y)
        self.tokenizer.fit_on_texts(X)
        self.tokenizer.word_index = {e: i for e, i in self.tokenizer.word_index.items() if i <= self.max_words}
        self.tokenizer.word_index[self.tokenizer.oov_token] = self.max_words + 1
        seqs = self._get_sequences(self._preprocess(X))
        self.model.fit([seqs], y, batch_size=self.bs, epochs=self.epochs, validation_split=0.3)

# import numpy as np
# import pandas as pd
# from sklearn.metrics import accuracy_score
# from keras.datasets import reuters
# from keras.preprocessing.sequence import pad_sequences
# from keras.utils.np_utils import to_categorical
# from keras.models import Sequential
# from keras.layers import Dense, SimpleRNN, Activation
# from keras import optimizers
# from keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.model_selection import train_test_split
#
# # pad the sequences with zeros
# # padding parameter is set to 'post' => 0's are appended to end of sequences
# #
# # X_train = pad_sequences(X_train, padding='post')
# # X_test = pad_sequences(X_test, padding='post')
#
# X_train = np.array(X_train).reshape((X_train.shape[0], X_train.shape[1], 1))
# X_test = np.array(X_test).reshape((X_test.shape[0], X_test.shape[1], 1))
#
# y_data = np.concatenate((y_train, y_test))
# y_data = to_categorical(y_data)
# y_train = y_data[:1395]
# y_test = y_data[1395:]
#
#
# def rnn():
#     model = Sequential()
#     model.add(SimpleRNN(50, input_shape=(49, 1), return_sequences=False))
#     model.add(Dense(46))
#     model.add(Activation('softmax'))
#
#     adam = optimizers.Adam(lr=0.001)
#     model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
#
#     return model
#
#
# model = KerasClassifier(build_fn=rnn, epochs=200, batch_size=50, verbose=1)
# model.fit(X_train, y_train)
#
# y_pred = model.predict(X_test)
# y_test_ = np.argmax(y_test, axis=1)
#
# print(accuracy_score(y_pred, y_test_))
