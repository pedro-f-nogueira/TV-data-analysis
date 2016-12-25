# coding=utf-8

"""
.. module:: model_decision_tree.py
    :synopsis: This module used a simple decision tree to determine several categories using program descriptions

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
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
from sklearn.metrics import accuracy_score
import cPickle as pickle
import subprocess


if __name__ == '__main__':

    df = util.get_df_from_conn('SELECT MASTER_CATEGORY, CLEAN_DESCR FROM T_TRAINING_TEXT_MODEL'
                               , 'local')

    x_train, x_test, y_train, y_test = util.get_train_test_target(df['CLEAN_DESCR'], df['MASTER_CATEGORY'])

    print "Creating the bag of words...\n"

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

    vocabulary = vectorizer.get_feature_names()

    vocabulary_encoded = []
    for x in vocabulary:
        vocabulary_encoded.append(x.encode('utf-8'))

    with open('vectorizer.pk', 'wb') as fin:
        pickle.dump(vectorizer, fin)

    # Numpy arrays are easy to work with, so convert the result to an array
    train_data_features = train_data_features.toarray()

    print train_data_features.shape

    print "Training the tree model..."

    # Fit the forest to the training set, using the bag of words
    simple_tree = DecisionTreeClassifier(min_samples_leaf=10, max_depth=15)
    simple_tree.fit(train_data_features, y_train)

    test_data_features = vectorizer.transform(x_test)
    test_data_features = test_data_features.toarray()

    print 'Saving the model for later usage...'

    with open('simple_tree.pk', 'wb') as fin:
        pickle.dump(simple_tree, fin)

    # simple_tree = joblib.load('simple_tree.pk')

    # forest = joblib.load('text_mining_forest.pkl')

    # Evaluating train results
    result_train = simple_tree.predict(train_data_features)
    print('Train accuracy: {}'.format(accuracy_score(y_train, result_train)))
    print metrics.classification_report(y_train, result_train)

    # Evaluating test results
    result_test = simple_tree.predict(test_data_features)
    print('Test accuracy: {}'.format(accuracy_score(y_test, result_test)))
    print metrics.classification_report(y_test, result_test)

    # dot_data = StringIO()
    export_graphviz(simple_tree, out_file='tree.dot', feature_names=vocabulary_encoded)

    subprocess.call(['dot', '-Tpdf', 'tree.dot', '-o' 'tree.pdf'])
