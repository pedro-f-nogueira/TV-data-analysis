# coding=utf-8

"""
.. module:: model_change_channel_bag.py
    :synopsis: This module uses a Adaboost with pre selected variables and bagging
               to predict the CHANNEL_CHANGE variable

    A Bagging classifier is an ensemble meta-estimator that fits base classifiers each on random subsets of the
    original dataset and then aggregate their individual predictions (either by voting or by averaging)
    to form a final prediction. Such a meta-estimator can typically be used as a way to reduce the variance of a
    black-box estimator (e.g., a decision tree), by introducing randomization into its construction procedure and
    then making an ensemble out of it.

.. moduleauthor:: Pedro Nogueira <pedro.fig.nogueira@gmail.com>
"""

import os
import sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)
del path


import utilities.utilities as util
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier


if __name__ == '__main__':
    print 'Fetching the dataframe...'

    df = util.get_df_from_conn('SELECT CHANNEL_CHANGE, RANK_OBJ_PROGRAM, RANK_PROGRAM_NAME_CHANNEL_CHANGE, '
                               'RANK_OBJ_CHANNEL, INFANTIL, CATEGORY_KIDS, DATA_SET, BOX_ID '
                               'FROM T_PYTHON_EVENTS_PRED '
                               'WHERE DATA_SET IN (1, 2) AND FLG_ZAPPING = 0 '
                               'ORDER BY ORIGIN_TIME', 'local')

    df['LAG_CHANNEL_CHANGE'] = df.groupby(['BOX_ID'])['CHANNEL_CHANGE'].shift(-1)

    df = df.drop('BOX_ID', axis=1)

    df["LAG_CHANNEL_CHANGE"].fillna(0, inplace=True)

    # Getting the train and test
    df_train = df[df['DATA_SET'] == 1]
    df_test = df[df['DATA_SET'] == 2]

    df_train = df_train.drop('DATA_SET', axis=1)
    df_test = df_test.drop('DATA_SET', axis=1)

    x_train = df_train.ix[:, 1:]
    x_test = df_test.ix[:, 1:]
    y_train = df_train['CHANNEL_CHANGE']
    y_test = df_test['CHANNEL_CHANGE']

    print 'Generating the model...'

    # Create and fit a decision tree
    seed = 7
    cart = DecisionTreeClassifier()
    num_trees = 400

    #Use bagging
    clf = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)

    print 'Fitting...'

    clf.fit(x_train, y_train)

    print 'Getting metrics...'

    util.evaluate_classifier(clf, x_train, y_train, x_test, y_test)
