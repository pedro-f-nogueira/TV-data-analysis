# coding=utf-8

"""
.. module:: model_bag_of_words.py
    :synopsis: This module writes the predictions done with the model trained with model_random_forest.py

.. moduleauthor:: Pedro Nogueira <pedro.fig.nogueira@gmail.com>
"""

import os
import sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)
del path

import utilities.utilities as util
from sklearn.externals import joblib

if __name__ == '__main__':

    print 'Fetching data...'

    df = util.get_df_from_conn('SELECT ID, CLEAN_DESCR FROM T_PYTHON_PROGRAMS_WITH_CATEGORY '
                               'WHERE LANGUAGE = \'PT\' AND MASTER_CATEGORY IS NULL'
                               , 'local')

    data = df['CLEAN_DESCR'].values

    print "Importing the vectorizer..."

    vectorizer = joblib.load('vectorizer.pk')

    train_data_features = vectorizer.transform(data)

    # Numpy arrays are easy to work with, so convert the result to an array
    train_data_features = train_data_features.toarray()

    print 'Importing the model...'

    forest = joblib.load('grid_search.pk')

    print 'Predicting the categories...'

    # Evaluating train results
    result = forest.predict(train_data_features)

    df['PREDICTED_CATEGORY'] = result

    print 'Writing the table to the database...'

    util.write_table_to_my_sql(df, 't_test_1', '', 'local')

    print 'Success'




