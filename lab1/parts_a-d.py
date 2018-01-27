"""
Curtis Crawford, 805024638
Abdullah-Al-Zubaer Imran, 

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

categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

trainset = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42);
testset = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42);

#plot the histogram for part a
#plt.hist(trainset.target, bins=range(min(trainset.target), (max(trainset.target) + 2)))
#plt.xlabel('Article target number')
#plt.ylabel('Number of articles target number')
#plt.title('Histogram of Article distribution, training set')
#plt.show()
#
#plt.hist(testset.target, bins=range(min(testset.target), (max(testset.target) + 2)))
#plt.xlabel('Article target number')
#plt.ylabel('Number of articles target number')
#plt.title('Histogram of Article distribution, testing set')
#plt.show()

ind = np.arange(8)
plt.hist(trainset.target, bins=range(min(trainset.target), (max(trainset.target))+2), width = 0.8)
plt.xlabel('Subclasses of Computer technology and Recreational activity')
plt.ylabel('Number of documents')
plt.xticks(ind, categories, fontsize = 10, rotation = 30, verticalalignment = 'top')
plt.title('Histogram of the number of training documents per class')

plt.figure()
plt.show()

plt.hist(testset.target, bins=range(min(testset.target), (max(testset.target))+2), width = 0.8)
plt.xlabel('Subclasses of Computer technology and Recreational activity')
plt.ylabel('Number of documents')
plt.xticks(ind, categories, fontsize = 10, rotation = 30, verticalalignment = 'top')
plt.title('Histogram of the number of testing documents per class')
plt.show()

#part b
#include the stemmer in the countvectorizer class
stemmer = nltk.stem.porter.PorterStemmer()
class stemCV(text.CountVectorizer):
    def build_analyzer(self):
        analyzer = super(stemCV, self).build_analyzer()
        return lambda doc: ([stemmer.stem(t) for t in analyzer(doc)])

vectorizer = stemCV(min_df=this_df, stop_words='english')

#tokenize, vectorize the stemmed data list
X = vectorizer.fit_transform(trainset.data)
testX = vectorizer.transform(testset.data)

#find TFxIDF
tfidf_transformer = text.TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X)
testX_tfidf = tfidf_transformer.transform(testX)
print "min_df=" + str(this_df) + " TFxIDF size: " + str(X_tfidf.shape)

#part d, part c is in its own script
from sklearn.decomposition import TruncatedSVD
lsi = TruncatedSVD(n_components=50)
X_lsi = lsi.fit_transform(X_tfidf)
testX_lsi = lsi.fit_transform(testX_tfidf)
print "min_df=" + str(this_df) + " after lsi: " + str(X_lsi.shape)
