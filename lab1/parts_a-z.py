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

#TODO
#partc, but only on training data
cset = fetch_20newsgroups(subset='train', shuffle=True, random_state=42);
print "PART C RESULTS TODO"

#part d
from sklearn.decomposition import TruncatedSVD
lsi = TruncatedSVD(n_components=50)
X_lsi = lsi.fit_transform(X_tfidf)
testX_lsi = lsi.fit_transform(testX_tfidf)
print "min_df=" + str(this_df) + " after lsi: " + str(X_lsi.shape)

#parte
#function to de-generalize to comp-tech or recreation
def e_class_combine(in_targets):
    out = []
    for i in in_targets:
        if i <= 3:
            out.append(0)
        else:
            out.append(1)
    return (out, ('comp-tech', 'rec'))

def analysis_res(real, pred, roc):
    acc = skl.metrics.accuracy_score(real, pred)
    rec = skl.metrics.precision_score(real, pred) 
    prc = skl.metrics.recall_score(real, pred)
    C =   skl.metrics.confusion_matrix(real, pred)

    print "\tAccuracy: " + str(acc)
    print "\tRecall: " + str(rec)
    print "\tPrecision: " + str(prc)
    print "\tConfusion Matrix: " + str(C)

    if (roc):
        print "still looking into ROC curve plotting..."
        score = [acc]
        roc = skl.metrics.roc_curve(real, score)

#convert data to only 2 classes
train_target,  target_names = e_class_combine(trainset.target)
test_target, target_names = e_class_combine(testset.target)

#first hard-margin
e_clf = skl.svm.LinearSVC(random_state=0, C=1000)
e_clf.fit(X_lsi, train_target)
res = e_clf.predict(testX_lsi)
#processing:
print "Part E Hard-margin results:"
analysis_res(test_target, res, False)

#now soft-margin
e_clf = skl.svm.LinearSVC(random_state=0, C=0.001)
e_clf.fit(X_lsi, train_target)
res = e_clf.predict(testX_lsi)
print "Part E Soft-margin results:"
analysis_res(test_target, res, False)

#partf, probably needs to be re-done to cross validate AND get recall/etc
for i in range(-3,4):
    c = 10 ** i
    f_clf = skl.svm.SVC(kernel='linear', C=c)
    scores = skl.model_selection.cross_val_score(f_clf, X_lsi, train_target, cv=5)
    print "k=" + str(i) + ": " + str(scores)

#partg
print "Cannot be done unless NNM is done, skipped for now"

#parth
