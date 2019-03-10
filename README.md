# The problem
Develop a system for plagiarism detection: you are given a set of textual documents and you
are asked to detect which documents contain significant overlapping in content.
# Initial thoughts
Detecting plagiarism in science papers shouldn't rely on purely comparing text and checking for matching sentences. But extracting the meaning from a text and comparing it to other such extractions should give us a much better insight in terms of quality.

So that's the basic idea: Can I somehow extract meaning (in form of ideas) from sentences and compare sets of those to sets from other texts and make an informed conclusion on wether there is plagiarism?

## Initial research
> The act of appropriating the literary composition of another author, or excerpts, ideas, or passages therefrom, and passing the material off as one's own creation.

Although my initial plan was to work with sentences, I decided to go for the most naive aproach and tokenize and vectorize all the texts given and find a way to compare those vectors.

# Data Set

Failed to find an easy way of downloading a set of papers from arxive.org and turning them into a usable .csv file, so I resorted to this:

https://www.kaggle.com/benhamner/nips-2015-papers#Papers.csv

Filtered out the columns that weren't of use and further reduced the set to 40 documents.

# Revised Plan Of Action

* Get appropriate dataset 
* Tokenize ALL texts (because we need to check if there are any overlaping texts in a given set) _Performance?_
* Remove stop words from the tokenized set
* Vectorize ALL the given texts
* Compare those vectors? _How? Machine learning with scikit-learn?_
## What's been done
* I managed to get a nice data set of scientific papers and import them in a easy-to-work-with format.
* Tokenized the texts in the data set using SpaCy
* Removed stopwords from the text using NLTK
* Tried to vectorize the token sets and compare vectors using Jaccard similarity, but couldn't test that because of memory errors.
## What still needs to be done
* Optimization of the tokenization and vectorization methods
* Finding a method of determining if two vectors are similar above a certain degree and mark them accordingly. Try if using Jaccard or Cosine similarity can help.

# Ways to improve it
* Removing numbers and other special markings (e.g. '[zlp+ 15]')
* Lemmatizing the tokenized texts
* Removing more stopwords
* Trying a better idea to compare vectors. The basic idea is to find or make a data set that has clearly marked pairs of similar texts and train a neural network with supervised learning to find similar texts. And then apply that knowledge to our test set.