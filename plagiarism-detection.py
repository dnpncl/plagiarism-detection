import pandas as pd
import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

spacy_nlp = spacy.load('en_core_web_sm')

# a set of plagiarized articles and the originals for ML training
trainSet = pd.read_csv('../data_sets/test.csv').apply(lambda x: x.astype(str).str.lower())

# the set needs to be in a single column 
# in order to vectorize based on the whole corpus
vectorizationSet = pd.DataFrame(columns=['Text'])
for i in trainSet.Source:
  vectorizationSet = vectorizationSet.append({'Text': i}, ignore_index=True)
for i in trainSet.Suspicious:
    vectorizationSet = vectorizationSet.append({'Text': i}, ignore_index=True)

# function for tokenization, lemmatization and removing stopwords
def tokenize(article):
  doc = spacy_nlp(article)
  lemmatized = ' '.join([token.lemma_ for token in doc])
  tokens = [token.text for token in spacy_nlp(lemmatized)]
  # remove stopwords
  stopWords = set(stopwords.words('english'))
  stopFree = [] 
  for w in tokens:
      if w not in stopWords:
          stopFree.append(w)

  return stopFree

# basic jaccard similarity
def getJaccardSim(str1, str2): 
    a = set(str1.split()) 
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

# pairing up cosine and jaccard similarities in a 2xN matrix
# and creating the 1xN label vector for the SGDClassifier
def pairUpCosJacc(matrix):
  res = []
  labels = []
  skipRange = len(trainSet.Source)
  #pair up actual source articles and their plagiates
  for i in range(skipRange):
    cos = cosine_similarity(matrix[i:i+1], matrix[i+skipRange:i+skipRange+1])
    jacc = getJaccardSim(trainSet.Source[i], trainSet.Suspicious[i])
    res.append([cos[0][0], jacc])
    labels.append(1)
  #pair up unplagiarized pairs
  for i in range(skipRange):
    if i==0:
      cos = cosine_similarity(matrix[i:i+1], matrix[2*skipRange-i-1:2*skipRange-i])
      jacc = getJaccardSim(trainSet.Source[i], trainSet.Suspicious[skipRange-1])
    else:
      cos = cosine_similarity(matrix[i:i+1], matrix[i+skipRange-1:i+skipRange])
      jacc = getJaccardSim(trainSet.Source[i], trainSet.Suspicious[i-1])
    res.append([cos[0][0], jacc])
    labels.append(0)
  return res, labels

# the scikit vectorizer creates vectors from the texts using the tokenize() function
tfidf = TfidfVectorizer(
    analyzer = 'word',
    tokenizer = tokenize,
    token_pattern = None)  

vectorizer = TfidfVectorizer()
tfidfMatrix = vectorizer.fit_transform(vectorizationSet.Text)

pairs, labels = pairUpCosJacc(tfidfMatrix)

# split the matrices into train and test sets
X_train, X_test, y_train, y_test = train_test_split(pairs, labels, test_size=0.33, random_state=42)

# train the classifier
clf = SGDClassifier(max_iter=1000, tol=1e-3)
clf.fit(X_train, y_train)

#make a prediction
prediction = clf.predict(X_test)

# compare the prediction to the test albel vector
print(accuracy_score(prediction, y_test))