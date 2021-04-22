from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM, Embedding
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.layers import Dense, Input, LSTM, Bidirectional, Activation, Conv1D, GRU, TimeDistributed
from keras.layers import Dropout, Embedding, GlobalMaxPooling1D, MaxPooling1D, Add, Flatten, SpatialDropout1D
from keras.layers import GlobalAveragePooling1D, BatchNormalization, concatenate
from keras.layers import Reshape, merge, Concatenate, Lambda, Average
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras.models import Model
from sklearn.base import BaseEstimator, TransformerMixin
from preprocess import cleaner
import numpy as np


class KerasLSTMClassifier(BaseEstimator, TransformerMixin):
  '''Wrapper class for keras text classification models that takes raw text as input.'''
  
  def __init__(self, max_words=30000, input_length=20, emb_dim=300, n_classes=3, epochs=15, batch_size=64, emb_idx=0):
    self.max_words = max_words
    self.input_length = input_length
    self.emb_dim = emb_dim
    self.n_classes = n_classes
    self.epochs = epochs
    self.bs = batch_size
    self.embeddings_index = emb_idx
    self.tokenizer = Tokenizer(num_words=self.max_words+1, lower=True, split=' ')
    self.model = self._get_model()
    
  def _get_model(self):
    input_text = Input((self.input_length,))
    text_embedding = Embedding(input_dim=self.max_words+1, output_dim=self.emb_dim, input_length=self.input_length, 
                               mask_zero=False, weights=[self.embeddings_index], trainable=False)(input_text)
    text_embedding = SpatialDropout1D(0.4)(text_embedding)
    #Bidirection LSTM
    bilstm =Bidirectional(LSTM(units=50,  recurrent_dropout=0.2, return_sequences = True))(text_embedding)
    #Dropout
    x = Dropout(0.2)(bilstm)
    # #LSTM
    # x =(LSTM(units=50,  recurrent_dropout=0.2, return_sequences = True))(x)
    # #Dropout
    # x = Dropout(0.2)(x)
    # #LSTM
    # x =(LSTM(units=50,  recurrent_dropout=0.2))(x)
    #Dense output layer
    out = Dense(units=self.n_classes, activation="softmax")(x)
    #Compile model, using adam optimizer
    model = Model(inputs=[input_text],outputs=[out])
    model.compile(optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"])
    return model
  
  def _get_sequences(self, texts):
    seqs = self.tokenizer.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=self.input_length, value=0)
  
  def _preprocess(self, texts):
    return [cleaner(x) for x in texts]
  #asfhjkahfkajhsfk
  def fit(self, X, y):
    '''Fit the vocabulary and the model.
       :params: X: list of texts. y: labels.
    '''
    self.tokenizer.fit_on_texts(X)
    self.tokenizer.word_index = {e: i for e,i in self.tokenizer.word_index.items() if i <= self.max_words}
    self.tokenizer.word_index[self.tokenizer.oov_token] = self.max_words + 1
    seqs = self._get_sequences(self._preprocess(X))
    self.model.fit([seqs ], y, batch_size=self.bs, epochs=self.epochs, validation_split=0.1)
  
  def predict_proba(self, X, y=None):
    seqs = self._get_sequences(self._preprocess(X))
    return self.model.predict(seqs)
  
  def predict(self, X, y=None):
    return np.argmax(self.predict_proba(X), axis=1)
  
  def score(self, X, y):
    y_pred = self.predict(X)
    return accuracy_score(np.argmax(y, axis=1), y_pred)