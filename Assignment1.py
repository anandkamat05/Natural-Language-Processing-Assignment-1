'''
Created on Sep 29, 2017

@author: Anand
'''

import nltk
import matplotlib
import random
import numpy as np
from sklearn import linear_model
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import confusion_matrix
from mpmath.tests.test_matrices import test_vector


positive_data = np.loadtxt(fname = 'rt-polarity.pos', dtype = "string",  delimiter = '\n')
negative_data = np.loadtxt(fname = 'rt-polarity.neg', dtype = "string",  delimiter = '\n')


#PreProcessing
pos_label = []
neg_label = []

raw_data = np.append(positive_data, negative_data)

for w in positive_data:
    pos_label.append('pos')
    neg_label.append('neg')
    
target = pos_label + neg_label

training_set = np.append(positive_data[:4500], negative_data[:4500])
test_set = np.append(positive_data[4500:5330], negative_data[4500:5330])

training_target = np.append(pos_label[:4500], neg_label[:4500])
test_target = np.append(pos_label[4500:5330], neg_label[4500:5330])

cv = CountVectorizer(analyzer='word', binary=False, decode_error='ignore'
        , input='content',lowercase=True, max_df=1.0, min_df=1,
        ngram_range=(1, 2), preprocessor=None, stop_words=None,
        strip_accents=None, tokenizer=None, vocabulary=None)


training_vector = cv.fit_transform(training_set) 
test_vector = cv.transform(test_set)
test_target_vector = cv.transform(test_target)

#Analysing using Naive Bayes
print("************************NAIVE BAYES THEORUM******************** \n")
clf_NB = MultinomialNB(alpha = 1.0, fit_prior=True, class_prior=None)
clf_NB.fit(training_vector, training_target)
prediction_NB = clf_NB.predict(test_vector)
print("Naive Bayes Prediction:")
print(prediction_NB)
print("\n Naive Bayes Accuracy:")
print(clf_NB.score(test_vector,test_target))
print("\n Naive Bayes Confusion Matrix:")
print(confusion_matrix(prediction_NB,test_target))

#Analyzing using SVM
print("\n ************************SUPPORT VECTOR MACHINE********************")
clf_SVM = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None)
clf_SVM.fit(training_vector, training_target)
print("SVM Prediction:")
prediction_SVM = clf_SVM.predict(test_vector)
print(prediction_SVM)
print("\n SVM Accuracy:")
print(clf_SVM.score(test_vector, test_target))
print("\n SVM Confusion Matrix:")
print(confusion_matrix(prediction_SVM,test_target))


#Analysing using logistic regression
print("\n ************************LOGITIC REGRESSION********************")
clf_logi = LogisticRegression(penalty= 'l2', tol=0.0001, C= 2.0, fit_intercept= True, intercept_scaling=1, class_weight=None, random_state = None)
clf_logi.fit(training_vector, training_target)
print("logistic regression Prediction:")
prediction_logi = clf_logi.predict(test_vector)
print(prediction_logi)
print("\n logistic regression Accuracy:")
print(clf_logi.score(test_vector, test_target))
print("\n logistic regression Confusion Matrix:")
print(confusion_matrix(prediction_logi,test_target))
