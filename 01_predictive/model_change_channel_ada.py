# coding=utf-8

"""
.. module:: model_change_channel_ada.py
    :synopsis: This module uses a Adaboost with pre selected variables
               to predict the CHANNEL_CHANGE variable

    The core principle of AdaBoost is to fit a sequence of weak learners (i.e., models that are only slightly better
    than random guessing, such as small decision trees) on repeatedly modified versions of the data. The predictions
    from all of them are then combined through a weighted majority vote (or sum) to produce the final prediction.

.. moduleauthor:: Pedro Nogueira <pedro.fig.nogueira@gmail.com>
"""

import os
import sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)
del path


import utilities.utilities as util
from sklearn.ensemble import AdaBoostClassifier


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

    # Create and fit an AdaBoosted decision tree
    seed = 7
    num_trees = 400

    # Create and fit an AdaBoosted decision tree
    clf = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)

    print 'Fitting...'

    clf.fit(x_train, y_train)

    print 'Getting metrics...'

    util.evaluate_classifier(clf, x_train, y_train, x_test, y_test)
