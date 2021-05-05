import re
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


#NLTK Stop words
stop_words_ = set(stopwords.words('english'))
#Custom Stop Words
my_sw = ['rt', 'ht', 'fb', 'amp', 'gt']
#WordNetLemma
wn = WordNetLemmatizer()

def slang_txt(token):
  if token == 'u':
    token = 'you'
  return  token not in stop_words_ and token not in list(string.punctuation) and token not in my_sw

def cleaner(word):
  #Decontracted words
  word = decontracted(word)
  #Remove users mentions
  word = re.sub(r'(@[^\s]*)', "", word)
  word = re.sub('[\W]', ' ', word)
  #Lemmatized
  list_word_clean = []
  for w1 in word.split(" "):
    if  slang_txt(w1.lower()):
      word_lemma =  wn.lemmatize(w1,  pos="v")
      list_word_clean.append(word_lemma)
  #Cleaning, lowering and remove whitespaces
  word = " ".join(list_word_clean)
  word = re.sub('[^a-zA-Z]', ' ', word)

  return word.lower().strip()

def decontracted(phrase):
    phrase = re.sub(r"https?://[A-Za-z0-9./]+", "url", phrase)
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"what's", "what is", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"j k", "jk", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase