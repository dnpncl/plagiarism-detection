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
* Tried to vectorize the token sets using tf-idf and compare vectors using Jaccard similarity, but couldn't test that because of memory errors I had no time to fix.
## What still needs to be done
* Optimization of the tokenization and vectorization methods
* Finding a method of determining if two vectors are similar above a certain degree and mark them accordingly. Try if using Jaccard or Cosine similarity can help.

# Ways to improve it
* Removing numbers and other special markings (e.g. '[zlp+ 15]')
* Lemmatizing the tokenized texts
* Removing more stopwords
* Trying a better idea to compare vectors. The basic idea is to find or make a data set that has clearly marked pairs of similar texts and train a neural network with supervised learning to find similar texts. And then apply that knowledge to our test set.

# Second phase
## New dataset
I found this dataset for figure plagiarism detection:

https://data.mendeley.com/datasets/gz3hztwm5p/1

and just used the text part of the data. It had just the source articles and plagiarized works based on them. The data was in .pdf format, so after turning it into text and making a convenient .csv file I had a table of Source and Suspicious pairs.
I used only 10 pairs (20 documents in total) to train and test the algorithm.

## Feature creation
I made the "tokenize" function which tokenizes, lemmatizes and removes stopwords from the articles. It is used in the sklearn vectorizer to create vectors of each text based on the whole corpus.

## Approach to ML
I learned of the difference between Jaccard and Cosine similarities and thought that cosine would be better suited for this particular problem but probably not as accurate on a smaller set of documents. So I thought of a combined approach. I would make a [cosine, jaccard]-looking feature and feed it to the ML part of the application where I'd have pairs of correlated articles labeled 1 and unrelated articles labeled 0. The algorithm should be able to learn from that feature set using a supervised learning approach.

The pair and label matrices were made in the "pairUpCosJacc" function, where firstly the similarities of the correct pairs of plagiates were computed and labeled as 1, and then the pairs were shifted to have pairs of texts without any contextual similarity labeled as 0.  

The problem being a simple classification, I opted for the SGDClassifier algorithm for no other reason than it being the first best option.

After spliting the set into train and test set (test size was 33% of the pairs), the classifier is trained and then a prediction is compared to the expected result.

## Results
* The algorithm labels pairs of texts correctly roughly 50% of the time (for a sample size of 10 pairs).
* The accuracy of the algorithm never goes under 50%.

## Possible improvements

* Using a larger sample from the data set (as the most obvious improvement)
* Using the optimized form of the Greedy String Tilling algorithm to compare texts.
* Removing the first page of the "Suspicious" texts as they were labeled by IEEE as plagiates. This doesn't change the meaning of the texts but by keeping it the algorithm might pair "suspicious" texts as plagiates of each other for having a similar first page.