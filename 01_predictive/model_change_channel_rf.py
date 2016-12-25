# coding=utf-8

"""
.. module:: model_change_channel_rf.py
    :synopsis: This module uses a Random Forest with Grid Search and pre selected variables
               to predict the CHANNEL_CHANGE variable

.. moduleauthor:: Pedro Nogueira <pedro.fig.nogueira@gmail.com>
"""

import os
import sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)
del path

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
import utilities.utilities as util
from sklearn import metrics


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

    forest = RandomForestClassifier(max_features='sqrt')

    parameter_grid = {
                      'max_depth': [4, 5, 6, 7, 8],
                      'n_estimators': [130, 150, 180, 200],
                      'criterion': ['gini', 'entropy']
                     }

    print 'Cross-validating...'

    cross_validation = StratifiedKFold(y_train, n_folds=5)

    print 'Performing grid search...'

    # The grid search provided by GridSearchCV exhaustively generates candidates from a grid of parameter values
    # specified with the param_grid parameter. For instance, the following param_grid:
    grid_search = GridSearchCV(forest,
                               param_grid=parameter_grid,
                               cv=cross_validation)

    print 'Fitting...'

    grid_search.fit(x_train, y_train)

    print 'Getting metrics...'

    util.evaluate_classifier_with_cv(grid_search, x_train, y_train, x_test, y_test)
