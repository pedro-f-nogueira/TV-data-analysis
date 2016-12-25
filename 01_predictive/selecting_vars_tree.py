# coding=utf-8

"""
.. module:: selecting_vars_tree.py
    :synopsis: This module uses features importances with a tree based classifier and with Lasso
               to determine the relevant variables in the CHANGE_CHANNEL problem.
               It alsos uses the pearson correlation to remove strongly correlated variables.

.. moduleauthor:: Pedro Nogueira <pedro.fig.nogueira@gmail.com>
"""
import os
import sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)
del path

from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
import utilities.utilities as util
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV


def return_importance(i_df, i_support):
    i = 0
    for column in list(i_df.columns.values):
        if i_support[i]:
            print(column)

        i = i + 1

if __name__ == '__main__':
    print 'Fetching the dataframe...'

    df = util.get_df_from_conn('SELECT CHANNEL_CHANGE, MORNING, AFTERNOON, EVENING, NIGHT, CHANNEL_N, GENERALISTAS, '
                               'INFORMACAO, ENTRETENIMENTO, DESPORTO, NACIONAIS, INFANTIL, OUTROS, SERIES, LIFESTYLE, '
                               'DOCUMENTARIOS, INTERNACIONAIS, ADULTOS, SEGUNDA, TERCA, QUARTA, QUINTA, SEXTA, SABADO, '
                               'DOMINGO, RANK_PROGRAM_NAME, RANK_CHANNEL_NAME, RANK_CHANNEL_GENRE,'
                               'RANK_CHANNEL_NAME_CHANNEL_CHANGE,RANK_CHANNEL_GENRE_CHANNEL_CHANGE, '
                               'RANK_PROGRAM_NAME_CHANNEL_CHANGE, FLG_IS_HD, RANK_OBJ_PROGRAM, RANK_OBJ_CHANNEL, '
                               'CLUSTER_KIDS, CLUSTER_GENERAL, CLUSTER_MUSIC, CLUSTER_SPORTS, CATEGORY_COOKING, '
                               'CATEGORY_DOCUMENTARIES, CATEGORY_MOVIES, CATEGORY_KIDS, CATEGORY_INFORMATION, '
                               'CATEGORY_MUSIC, CATEGORY_SOAP_OPERAS, CATEGORY_SERIES, CATEGORY_CONTESTS, '
                               'CATEGORY_ADULTS, CATEGORY_OTHERS, BOX_ID, DATA_SET '
                               'FROM T_PYTHON_EVENTS_PRED '
                               'WHERE DATA_SET IN (1, 2) AND FLG_ZAPPING = 0 ORDER BY ORIGIN_TIME', 'local')

    df['LAG_CHANNEL_CHANGE'] = df.groupby(['BOX_ID'])['CHANNEL_CHANGE'].shift(-1)

    df = df.drop('BOX_ID', axis=1)

    print 'Getting Pearson Correlation...'

    # Getting the train and test
    df_train = df[df['DATA_SET'] == 1]
    df_test = df[df['DATA_SET'] == 2]

    # Finding correlations between variables using Pearson correlation

    print util.get_pearson_correlation(df_train)
    """
    These features have correlations above 0.8
    Remove                Keep
    FLG_IS_HD             SERIES
    CHANNEL_N             RANK_CHANNEL_NAME_CHANNEL_CHANGE
    RANK_PROGRAM_NAME     RANK_CHANNEL_NAME_CHANNEL_CHANGE
    RANK_CHANNEL_GENRE    RANK_CHANNEL_GENRE_CHANNEL_CHANGE
    """

    # Dropping correlated features
    df_train = df_train.drop('FLG_IS_HD', axis=1)
    df_train = df_train.drop('CHANNEL_N', axis=1)
    df_train = df_train.drop('RANK_PROGRAM_NAME', axis=1)
    df_train = df_train.drop('RANK_CHANNEL_GENRE', axis=1)
    df_train = df_train.drop('RANK_CHANNEL_NAME', axis=1)
    df_train = df_train.drop('RANK_CHANNEL_NAME_CHANNEL_CHANGE', axis=1)

    df_test = df_test.drop('FLG_IS_HD', axis=1)
    df_test = df_test.drop('CHANNEL_N', axis=1)
    df_test = df_test.drop('RANK_PROGRAM_NAME', axis=1)
    df_test = df_test.drop('RANK_CHANNEL_GENRE', axis=1)
    df_test = df_test.drop('RANK_CHANNEL_NAME', axis=1)
    df_test = df_test.drop('RANK_CHANNEL_NAME_CHANNEL_CHANGE', axis=1)

    print 'Getting Pearson Correlation x2...'

    print util.get_pearson_correlation(df_train)

    x_train = df_train.ix[:, 1:]
    x_test = df_test.ix[:, 1:]
    y_train = df_train['CHANNEL_CHANGE']
    y_test = df_test['CHANNEL_CHANGE']

    print 'Finding features importances using a tree based estimator...'

    # Tree-based estimators can be used to compute feature importances, which in turn can be used to
    # discard irrelevant features
    clf = ExtraTreesClassifier(n_estimators=200)
    clf = clf.fit(x_train, y_train)

    # We use the base estimator LassoCV since the L1 norm promotes sparsity of features.
    clf_1 = LassoCV()
    clf_1 = clf_1.fit(x_train, y_train)

    features = pd.DataFrame()
    features['feature'] = x_train.columns
    features['importance'] = clf.feature_importances_

    print 'Printing the tree selected variables...'

    print features.sort_values(by='importance', ascending=False)
    model = SelectFromModel(clf, prefit=True)
    train_new = model.transform(x_train)

    print 'Getting the data shape...'
    print train_new.shape

    print 'Printing the Lasso selected variables...'

    model = SelectFromModel(clf, prefit=True)
    train_new = model.transform(x_train)

    print 'Getting the data shape...'

    print return_importance(x_train, model.get_support(indices=False))
