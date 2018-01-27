"""
Curtis Crawford, 805024638
Abdullah-Al-Zubaer Imran, 804733867 

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

categories = ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'misc.forsale', 'soc.religion.christian']
trainset = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42);
testset = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42);

#include the stemmer in the countvectorizer class
stemmer = nltk.stem.porter.PorterStemmer()
class stemCV(text.CountVectorizer):
    def build_analyzer(self):
        analyzer = super(stemCV, self).build_analyzer()
        return lambda doc: ([stemmer.stem(t) for t in analyzer(doc)])
vectorizer = stemCV(min_df=this_df, stop_words='english')

#print testset.target[1:20]

#processing, pipelines to get predict_proba:
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.decomposition import NMF
from sklearn.naive_bayes import MultinomialNB
pipe = Pipeline([
#    ('vect', stemCV(min_df=this_df, stop_words='english')),
    ('tfidf', text.TfidfTransformer()),
    ('reduce_dim', TruncatedSVD(n_components=50)),
    ('clf', SVC(decision_function_shape='ovo', kernel='linear')),
])


#analysis
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score

def fit_predict_plot_roc(pipe, train_data, train_label, test_data, test_label):
    pipe.fit(train_data, train_label)
    
    pred_res = pipe.predict(test_data)
    print "Confusion matrix: "
    print str(confusion_matrix(test_label, pred_res))
    print "Accuracy:  " + str(accuracy_score(test_label, pred_res))
    print "Precision: " + str(precision_score(test_label, pred_res, average="micro"))
    print "Recall:    " + str(recall_score(test_label, pred_res, average="micro"))
    print "-----------------------------------"
    print "\n"
    return pipe


#tokenize, vectorize the stemmed data list
traindata = vectorizer.fit_transform(trainset.data)
testdata = vectorizer.transform(testset.data)

print "One against one:"
fit_predict_plot_roc(pipe, traindata, trainset.target, testdata, testset.target)

print "One against all:"
new_clf = LinearSVC()
pipe.set_params(clf=new_clf)
fit_predict_plot_roc(pipe, traindata, trainset.target, testdata, testset.target)



#do Naive bayes classification
print "Naive Bayes classifier and NMF decomposition:"
new_reduce = NMF(n_components=50, init='random', random_state=0)
new_clf = MultinomialNB()
pipe.set_params(clf=new_clf, reduce_dim=new_reduce)
fit_predict_plot_roc(pipe, traindata, trainset.target, testdata, testset.target)

