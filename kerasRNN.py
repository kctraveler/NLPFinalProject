import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, SimpleRNN, Activation, InputLayer, Embedding
from keras.models import Sequential
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from preprocess import cleaner


class KerasRNNClassifier:

    def __init__(self, max_words=30000, input_length=20, emb_dim=300, n_classes=3, epochs=20, batch_size=64, emb_idx=0):
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
        model.add(InputLayer((self.input_length,)))
        model.add(
            Embedding(
                input_dim=self.max_words, output_dim=self.emb_dim,
                input_length=self.input_length, mask_zero=False,
                trainable=False))
        model.add(SimpleRNN(5, return_sequences=False))
        model.add(Dense(3))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

        model.summary()
        return model

    def _get_sequences(self, texts):
        seqs = self.tokenizer.texts_to_sequences(texts)
        seqs = pad_sequences(seqs, maxlen=self.input_length, value=0)
        return seqs

    def _predict_probability(self, X, y=None):
        seqs = self._get_sequences(self._preprocess(X))
        return self.model.predict(seqs)

    def _predict(self, X, y=None):
        return np.argmax(self._predict_probability(X), axis=1)

    def fit(self, X, y):
        y = to_categorical(y)
        self.tokenizer.fit_on_texts(X)
        self.tokenizer.word_index = {e: i for e, i in self.tokenizer.word_index.items() if i <= self.max_words}
        self.tokenizer.word_index[self.tokenizer.oov_token] = self.max_words + 1
        seqs = self._get_sequences(self._preprocess(X))
        self.model.fit([seqs], y, batch_size=self.bs, epochs=self.epochs, validation_split=0.3)
        history = self.model.fit([seqs ], y, batch_size=self.bs, epochs=self.epochs, validation_split=0.1)
        return history

    def score(self, X, y):
        y_pred = self.predict(X)
        print('Confusion Matrix:')
        print(confusion_matrix(y, y_pred))
        print('Classification Report:')
        print(classification_report(y, y_pred))
        return accuracy_score(y, y_pred)
