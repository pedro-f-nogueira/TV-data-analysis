# coding=utf-8

"""
.. module:: model_user_rec.py
    :synopsis: This module will give recommendations for programs based on user info
               This module will ony run using graphlab in a Jupyter notebook

.. moduleauthor:: Pedro Nogueira <pedro.fig.nogueira@gmail.com>
"""

import graphlab as gl
import sqlalchemy as sql
import pandas as pd
import matplotlib


def get_df_from_conn(i_query, i_place):
    """
    This function creates a connection to the database
    Returns a df according to the input query

    i_query -> query to be returned to the dataframe
    """

    if i_place == 'aws':
        config = ''
    elif i_place == 'remote':
        config = 'deleted'
    else:
        config = 'deleted'

    engine = sql.create_engine(
        config,
        pool_size=100,
        pool_recycle=3600,
    )
    db = engine.connect()
    my_df = pd.read_sql(i_query, con=db)
    db.close()

    return my_df


if __name__ == '__main__':
    print 'Fetching records...'

    df = get_df_from_conn('SELECT BOX_ID, PROGRAM_NAME, DIVISE_RATING FROM T_PROG_RECOMMENDATION_TOTAL;', 'local')

    train_data = gl.SFrame(df)

    gl.canvas.set_target('ipynb')

    training_subset, validation_subset = gl.recommender.util.random_split_by_user(train_data,
                                                                                  user_id="BOX_ID",
                                                                                  item_id="PROGRAM_NAME",
                                                                                  max_num_users=100,
                                                                                  item_test_proportion=0.3)

    model_0 = gl.popularity_recommender.create(training_subset, user_id="BOX_ID", item_id="PROGRAM_NAME")

    model_1 = gl.recommender.item_similarity_recommender.create(training_subset, user_id="BOX_ID",
                                                                item_id="PROGRAM_NAME")

    # RankingFactorizationRecommender trains a model capable of predicting a score for each possible combination of users
    # and items. The internal coefficients of the model are learned from known scores of users and items.
    # Recommendations are then based on these scores.
    model_2 = gl.recommender.ranking_factorization_recommender.create(training_subset, user_id="BOX_ID",
                                                                      item_id="PROGRAM_NAME",
                                                                      target="DIVISE_RATING", ranking_regularization=1)

    # Manually evaluate on the validation data. Evaluation results contain per-user, per-item and overall RMSEs.
    results = gl.recommender.util.compare_models(validation_subset, [model_0, model_1, model_2],
                                                 metric='precision_recall',
                                                 exclude_known_for_precision_recall=True,
                                                 make_plot=True)

    model_0.show(view='Evaluation')

    model_1.show(view='Evaluation')

    model_2.show(view='Evaluation')