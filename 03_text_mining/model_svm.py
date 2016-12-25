# coding=utf-8

"""
.. module:: model_svm.py
        :synopsis: This module used a SVM to determine several categories using program descriptions
                   It uses data that was categorized by humans in order to train the algorythm. This data is composed
                   by a category of a given program and its clean description.
                   It may take a few hours to run.

.. moduleauthor:: Pedro Nogueira <pedro.fig.nogueira@gmail.com>
"""

import os
import sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)
del path

import utilities.utilities as util
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import accuracy_score
import cPickle as pickle

if __name__ == '__main__':

    df = util.get_df_from_conn('SELECT MASTER_CATEGORY, CLEAN_DESCR FROM T_TRAINING_TEXT_MODEL'
                               , 'local')

    # Performing stemming
    df['STEMMED_DESCR'] = df.CLEAN_DESCR.apply(util.perform_stemming)

    # Performing lemmatization
    df['LEMMA_DESCR'] = df.CLEAN_DESCR.apply(util.perform_lemmatization)

    x_train, x_test, y_train, y_test = util.get_train_test_target(df['LEMMA_DESCR'], df['MASTER_CATEGORY'])

    print "Creating the CountVectorizer...\n"

    # Initialize the "CountVectorizer" object, which is scikit-learn's
    vectorizer = CountVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 max_features=5000)

    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of
    # strings.
    train_data_features = vectorizer.fit_transform(x_train)

    # Lets apply tfiff to the output of countvectorizer
    tfidf_transformer = TfidfTransformer()
    train_data_features = tfidf_transformer.fit_transform(train_data_features)

    with open('vectorizer.pk', 'wb') as fin:
        pickle.dump(vectorizer, fin)

    # Numpy arrays are easy to work with, so convert the result to an array
    train_data_features = train_data_features.toarray()

    print "Training the SVM..."

    classifier_rbf = OneVsRestClassifier(SVC(kernel='rbf'))

    classifier_rbf.fit(train_data_features, y_train)

    test_data_features = vectorizer.transform(x_test)
    test_data_features = tfidf_transformer.transform(test_data_features)
    test_data_features = test_data_features.toarray()

    print 'Saving the model for later usage...'

    with open('classifier_rbf.pk', 'wb') as fin:
        pickle.dump(classifier_rbf, fin)

    #classifier_rbf = joblib.load('classifier_rbf.pkl')

    # Evaluating train results
    result_train = classifier_rbf.predict(train_data_features)
    print('Train accuracy: {}'.format(accuracy_score(y_train, result_train)))
    print metrics.classification_report(y_train, result_train)

    # Evaluating test results
    result_test = classifier_rbf.predict(test_data_features)
    print('Test accuracy: {}'.format(accuracy_score(y_test, result_test)))
    print metrics.classification_report(y_test, result_test)
