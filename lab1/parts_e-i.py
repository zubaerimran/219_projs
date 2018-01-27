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

categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

raw_trainset = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42);
raw_testset = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42);

#include the stemmer in the countvectorizer class
stemmer = nltk.stem.porter.PorterStemmer()
class stemCV(text.CountVectorizer):
    def build_analyzer(self):
        analyzer = super(stemCV, self).build_analyzer()
        return lambda doc: ([stemmer.stem(t) for t in analyzer(doc)])

#because the stemmer takes SO long in pipeline
vectorizer = stemCV(min_df=this_df, stop_words='english')

#tokenize, vectorize the stemmed data list
trainset = vectorizer.fit_transform(raw_trainset.data)
testset = vectorizer.transform(raw_testset.data)

#function to de-generalize to comp-tech or recreation
def e_class_combine(in_targets):
    out = []
    for i in in_targets:
        if i <= 3:
            out.append(0)
        else:
            out.append(1)
    return (out, ('comp-tech', 'rec'))

#plot and run pipeline functions from the discussion section
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score

def plot_roc(fpr, tpr):
    fig, ax = plt.subplots()

    roc_auc = auc(fpr,tpr)

    ax.plot(fpr, tpr, lw=2, label= 'area under curve = %0.4f' % roc_auc)

    ax.grid(color='0.7', linestyle='--', linewidth=1)

    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate',fontsize=15)
    ax.set_ylabel('True Positive Rate',fontsize=15)

    ax.legend(loc="lower right")

    for label in ax.get_xticklabels()+ax.get_yticklabels():
        label.set_fontsize(15)

    return

def print_stats(test_label, pred_res):
    print "Confusion matrix: "
    print str(confusion_matrix(test_label, pred_res))
    print "Accuracy:  " + str(accuracy_score(test_label, pred_res))
    print "Precision: " + str(precision_score(test_label, pred_res))
    print "Recall:    " + str(recall_score(test_label, pred_res))
    print "-----------------------------------"
    print "\n"
    return


global_roc = []
def fit_predict_plot_roc(pipe, train_data, train_label, test_data, test_label):
    pipe.fit(train_data, train_label)
    
    pred_res = pipe.predict(test_data)
    prob_score = pipe.predict_proba(test_data)
    fpr, tpr, _ = roc_curve(test_label, prob_score[:,1])
    global_roc.append((fpr, tpr))
#    plot_roc(fpr, tpr)
    print_stats(test_label, pred_res)
    return pipe


#convert data to only 2 classes
train_target,  target_names = e_class_combine(raw_trainset.target)
test_target, target_names = e_class_combine(raw_testset.target)

#processing, pipelines to get predict_proba:
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
pipe = Pipeline([
#    ('vect', stemCV(min_df=this_df, stop_words='english')),
    ('tfidf', text.TfidfTransformer()),
    ('reduce_dim', TruncatedSVD(n_components=50)),
    ('clf', SVC(kernel='linear', C=1000, probability=True)),
])

print "Part E heavy:"
fit_predict_plot_roc(pipe, trainset, train_target, testset, test_target)

print "Part E soft:"
new_clf = SVC(kernel='linear', C=0.001, probability=True)
pipe.set_params(clf=new_clf)
fit_predict_plot_roc(pipe, trainset, train_target, testset, test_target)

#part f use built in cross val to generate predictions too!
for i in range(-3, 3):
    from sklearn.model_selection import cross_val_predict
    new_clf = SVC(kernel='linear', C=(10**i), probability=True)
    pipe.set_params(clf=new_clf)
    pred_res = cross_val_predict(pipe, trainset, train_target, cv=5)
    print "Cross validation results for k=" + str(i)
    print_stats(train_target, pred_res)

#part g cannot be done without NNM reduction instead of LSI
from sklearn.decomposition import NMF
from sklearn.naive_bayes import MultinomialNB
new_reduce = NMF(n_components=50, init='random', random_state=0)
new_clf = MultinomialNB()
pipe.set_params(clf=new_clf, reduce_dim=new_reduce)
print "Naive Bayes classifier and NMF decomposition:"
fit_predict_plot_roc(pipe, trainset, train_target, testset, test_target)

#part h
from sklearn.linear_model import LogisticRegression
new_reduce = TruncatedSVD(n_components=50)
new_clf = LogisticRegression(penalty='l1')
pipe.set_params(clf=new_clf, reduce_dim=new_reduce)
print "Logistic Regression with l2 penalty:"
fit_predict_plot_roc(pipe, trainset, train_target, testset, test_target)


#part i
new_clf = LogisticRegression(penalty='l1')
pipe.set_params(clf=new_clf, reduce_dim=new_reduce)
print "Logistic Regression with l1 penalty:"
fit_predict_plot_roc(pipe, trainset, train_target, testset, test_target)

for roc in global_roc:
    plt.plot(roc[0], roc[1], lw=2)

plt.grid(color='0.7', linestyle='--', linewidth=1)

#plt.axis([-0.1, 1.1, 0.0, 1.05])
plt.axis([0.0, 0.2, 0.7, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.legend(['Heavy-margin', 'soft-margin', 'Naive Bayes', 'logistic regression l2', 'logistic regression l1'], loc="lower right")
plt.show()
