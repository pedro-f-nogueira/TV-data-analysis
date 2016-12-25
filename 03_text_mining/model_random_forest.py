# coding=utf-8

"""
.. module:: model_random_forest.py
    :synopsis: This module used a Random Forest to determine several categories using program descriptions
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
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
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

    # Lets apply tfiff to the output of countvectorizer
    tfidf_transformer = TfidfTransformer()
    train_data_features = tfidf_transformer.fit_transform(train_data_features)

    with open('vectorizer.pk', 'wb') as fin:
        pickle.dump(vectorizer, fin)

    # Numpy arrays are easy to work with, so convert the result to an array
    train_data_features = train_data_features.toarray()

    print "Training the random forest..."

    forest = RandomForestClassifier(max_features='sqrt')

    print 'Cross-validating...'

    cross_validation = StratifiedKFold(y_train, n_folds=5)

    print 'Performing grid search...'

    # The grid search provided by GridSearchCV exhaustively generates candidates from a grid of parameter values
    # specified with the param_grid parameter. For instance, the following param_grid:
    parameter_grid = {
        'max_depth': [200, 300],
        'n_estimators': [140, 160],
        'criterion': ['gini', 'entropy']
    }

    grid_search = GridSearchCV(forest,
                               param_grid=parameter_grid,
                               cv=cross_validation)

    # Fit the forest to the training set
    grid_search.fit(X=train_data_features, y=y_train)

    test_data_features = vectorizer.transform(x_test)
    test_data_features = tfidf_transformer.transform(test_data_features)
    test_data_features = test_data_features.toarray()

    print 'Saving the model for later usage...'

    with open('grid_search.pk', 'wb') as fin:
        pickle.dump(grid_search, fin)

    #forest = joblib.load('text_mining_forest.pkl')

    # Evaluating train results
    result_train = grid_search.predict(train_data_features)
    print('Train accuracy: {}'.format(accuracy_score(y_train, result_train)))
    print metrics.classification_report(y_train, result_train)

    # Evaluating test results
    result_test = grid_search.predict(test_data_features)
    print('Test accuracy: {}'.format(accuracy_score(y_test, result_test)))
    print metrics.classification_report(y_test, result_test)
