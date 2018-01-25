"""
Curtis Crawford, 805024638
PARTNER?

EE219, Winter 2018
Project 1
"""

import numpy as np
import sklearn as skl
from sklearn.datasets import fetch_20newsgroups
import sklearn.datasets as skd
from sklearn.feature_extraction import text
import matplotlib as mpl
import matplotlib.pyplot as plt
import nltk
import sys

if len(sys.argv) != 2:
    print "Script must be called with a value for min_dif as only argument"
    exit()

this_df = int(sys.argv[1])

print "This run will use min_df=" + str(this_df)

print "numpy version: " + np.__version__
print "sklearn version: " + skl.__version__
print "matplotlib version: " + mpl.__version__
print "nltk version: " + nltk.__version__

fullset = fetch_20newsgroups(shuffle=True, random_state=42);
trainset = fetch_20newsgroups(subset='train', shuffle=True, random_state=42);
testset = fetch_20newsgroups(subset='test', shuffle=True, random_state=42);

cset = [' ']*20
print max(fullset.target)
print len(cset)
#convert C into a list of 20 total values
for i in range(0, len(fullset.target)):
    target = fullset.target[i]
    data = fullset.data[i]
    cur = cset[target]
    cset[target] = cur + " " + data

#include the stemmer in the countvectorizer class
stemmer = nltk.stem.porter.PorterStemmer()
class stemCV(text.CountVectorizer):
    def build_analyzer(self):
        analyzer = super(stemCV, self).build_analyzer()
        return lambda doc: ([stemmer.stem(t) for t in analyzer(doc)])

vectorizer = stemCV(min_df=this_df, stop_words='english')

#tokenize, vectorize the stemmed data list
X = vectorizer.fit_transform(cset)

#find TFxIDF
tfidf_transformer = text.TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X)

#print out 10 largest TFxICF from the four categories
for cat in ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'misc.forsale', 'soc.religion.christian']:
    idx = fullset.target_names.index(cat)
    print cat + ":"
    row = X_tfid.toarray()[idx]
    for i in range(10):
        idx_mc = max(row)
        print str(idx_mc)
        most_common = vectorizer.get_feature_names()[idx_mc]
        print str(i) + "th most common item is: " + str(most_common)
        row[most_com] = min(row)
