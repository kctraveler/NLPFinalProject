import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from zipfile import ZipFile
import re
from nltk.stem.snowball import SnowballStemmer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
from sklearn.linear_model import SGDClassifier
import logging
from textblob import TextBlob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import warnings;warnings.filterwarnings('ignore')
import preprocess
from kerasGRU import KerasGRUClassifier
from kerasLSTM import KerasLSTMClassifier
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0],True)


hate_speech_corpus = pd.read_csv("hate_speech.csv")
# #Shape
# print("Shape: ", hate_speech_corpus.shape)
# #Class distribution
# print("Class Values:\n", hate_speech_corpus['class'].value_counts())

hate_speech_corpus_final = hate_speech_corpus[['class', 'tweet']]

X = hate_speech_corpus_final[['tweet']]
y = hate_speech_corpus_final[['class']]
encoder = LabelEncoder()
y = encoder.fit_transform(y)
Y = np_utils.to_categorical(y)
X = X.values
X = [x[0] for x in X]
sns.barplot(['Non Toxic', 'Toxic', 'Hate'], hate_speech_corpus_final['class'].map({0:"Non Toxic", 1: "Toxic", 2: "Hate"}).value_counts(), palette="icefire")
plt.title('Count of Toxic and Hate Comments of Dataset')
plt.show()

#Testing cleaner
# for idx in hate_speech_corpus_final.tail(15).index:
#   print(preprocess.cleaner(hate_speech_corpus_final.iloc[idx]['tweet']),'\n'  , hate_speech_corpus_final.iloc[idx]['tweet'], idx)
#   print("************")

import spacy
from keras.preprocessing.text import Tokenizer
#!python -m spacy download en_core_web_lg
nlp = spacy.load("en_core_web_lg")


#Embedding
tokenizer = Tokenizer(num_words=30000)
tokenizer.fit_on_texts(X)
embeddings_index = np.zeros((30000 + 1, 300))
for word, idx in tokenizer.word_index.items():
    try:
          embedding = nlp.vocab[word].vector
          embeddings_index[idx] = embedding
    except:
      pass
print(embeddings_index[1])
#lstmMODEL = KerasLSTMClassifier()


x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 444, stratify=y)
# lstmMODEL = KerasLSTMClassifier(emb_idx= embeddings_index)
# print(lstmMODEL.model.summary())
# lstmMODEL.fit(x_train, y_train)
# print(lstmMODEL.score(x_test, y_test))

gru = KerasGRUClassifier()

gru.fit(x_train,y_train)
print(gru.score(x_test,y_test))