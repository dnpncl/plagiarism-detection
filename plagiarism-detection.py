import pandas as pd
import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

spacy_nlp = spacy.load('en_core_web_sm')

# read the data set in a nice format
dtSet = pd.read_csv('Papers.csv').apply(lambda x: x.astype(str).str.lower())
# reduce the data set to 40 texts with just two columns
restrictedSet = dtSet.filter(['Title', 'PaperText'])[:40]

# function for tokenization and removing stopwords
def tokenize(article):
  doc = spacy_nlp(article)
  tokens = [token.text for token in doc]
  # remove stopwords
  stopWords = set(stopwords.words('english'))
  stopFree = [] 
  for w in tokens:
      if w not in stopWords:
          stopFree.append(w)
  return stopFree

# turn the texts into sets of tokens

tokenizedFrame = pd.DataFrame(columns=['Title', 'PaperTokenized'])
for row in restrictedSet.itertuples():
  token = tokenize(row.PaperText)
  tokenFrame = pd.DataFrame([[row.Title, token]], columns=['Title', 'PaperTokenized'])
  tokenizedFrame = tokenizedFrame.append(tokenFrame)

tfidf = TfidfVectorizer(
    analyzer='word',
    tokenizer=tokenize,
    preprocessor=tokenize,
    token_pattern=None)  

tfidf.fit(tokenizedFrame['PaperTokenized'])

#TODO: train neural network to find similar texts (above 50% similarity)