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
import warnings;warnings.filterwarnings('ignore')
import preprocess
#from kerasLSTM import KerasLSTMClassifier

hate_speech_corpus = pd.read_csv("hate_speech.csv")
#Shape
print("Shape: ", hate_speech_corpus.shape)
#Class distribution
print("Class Values:\n", hate_speech_corpus['class'].value_counts())

hate_speech_corpus_final = hate_speech_corpus[['class', 'tweet']]

print(hate_speech_corpus_final[0:5])

X = hate_speech_corpus_final[['tweet']]
y = hate_speech_corpus_final[['class']]

sns.barplot(['Non Toxic', 'Toxic', 'Hate'], hate_speech_corpus_final['class'].map({0:"Non Toxic", 1: "Toxic", 2: "Hate"}).value_counts(), palette="icefire")
plt.title('Count of Toxic and Hate Comments of Dataset')
plt.show()

#Testing cleaner
for idx in hate_speech_corpus_final.tail(15).index:
  print(preprocess.cleaner(hate_speech_corpus_final.iloc[idx]['tweet']),'\n'  , hate_speech_corpus_final.iloc[idx]['tweet'], idx)
  print("************")

import spacy
from keras.preprocessing.text import Tokenizer
#!python -m spacy download en_core_web_lg
nlp = spacy.load("en_core_web_lg")
#Embedding
tokenizer = Tokenizer(num_words=30000)
tokenizer = Tokenizer(num_words=30000)
tokenizer.fit_on_texts(hate_speech_corpus_final['tweet'])
embeddings_index = np.zeros((30000 + 1, 300))
for word, idx in tokenizer.word_index.items():
    try:
          embedding = nlp.vocab[word].vector
          embeddings_index[idx] = embedding
    except:
      pass
print(embeddings_index[1])
#lstmMODEL = KerasLSTMClassifier()