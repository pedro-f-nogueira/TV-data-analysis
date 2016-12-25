# coding=utf-8

"""
.. module:: gen_t_python_events.py
    :synopsis: This module generates data to be used for visualization of the CHANNEL_CHANGE problem variables

.. moduleauthor:: Pedro Nogueira <pedro.fig.nogueira@gmail.com>
"""
import os
import sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)
del path

import utilities.utilities as util
import pandas as pd


if __name__ == '__main__':
    print 'Fetching records...'

    df = util.get_df_from_conn('SELECT * FROM V_EVENTS '
                               'ORDER BY ORIGIN_TIME;', 'local')

    print 'Calculating features step one...'

    # Calculating the CHANNEL_GENRE in binary
    df['Generalistas'] = df.CHANNEL_GENRE.apply(util.get_channel_genre, args=(1,))
    df['Informacao'] = df.CHANNEL_GENRE.apply(util.get_channel_genre, args=(2,))
    df['Entretenimento'] = df.CHANNEL_GENRE.apply(util.get_channel_genre, args=(3,))
    df['Desporto'] = df.CHANNEL_GENRE.apply(util.get_channel_genre, args=(4,))
    df['Nacionais'] = df.CHANNEL_GENRE.apply(util.get_channel_genre, args=(5,))
    df['Infantil'] = df.CHANNEL_GENRE.apply(util.get_channel_genre, args=(6,))
    df['Outros'] = df.CHANNEL_GENRE.apply(util.get_channel_genre, args=(7,))
    df['Series'] = df.CHANNEL_GENRE.apply(util.get_channel_genre, args=(8,))
    df['Lifestyle'] = df.CHANNEL_GENRE.apply(util.get_channel_genre, args=(9,))
    df['Documentarios'] = df.CHANNEL_GENRE.apply(util.get_channel_genre, args=(10,))
    df['Internacionais'] = df.CHANNEL_GENRE.apply(util.get_channel_genre, args=(11,))
    df['Adultos'] = df.CHANNEL_GENRE.apply(util.get_channel_genre, args=(12,))

    # Calculating the hour to periods
    df['PERIOD_OF_DAY'] = df.HOUR.apply(util.get_period_of_day)

    df['MORNING'] = df.PERIOD_OF_DAY.apply(util.get_period_of_day_binary, args=(1,))
    df['AFTERNOON'] = df.PERIOD_OF_DAY.apply(util.get_period_of_day_binary, args=(2,))
    df['EVENING'] = df.PERIOD_OF_DAY.apply(util.get_period_of_day_binary, args=(3,))
    df['NIGHT'] = df.PERIOD_OF_DAY.apply(util.get_period_of_day_binary, args=(4,))

    # Calculating the day of the week
    df['DAY_OF_WEEK'] = df.ORIGIN_TIME.apply(util.get_day_of_week)

    df['SEGUNDA'] = df.DAY_OF_WEEK.apply(util.get_day_of_week_binary, args=(1,))
    df['TERCA'] = df.DAY_OF_WEEK.apply(util.get_day_of_week_binary, args=(2,))
    df['QUARTA'] = df.DAY_OF_WEEK.apply(util.get_day_of_week_binary, args=(3,))
    df['QUINTA'] = df.DAY_OF_WEEK.apply(util.get_day_of_week_binary, args=(4,))
    df['SEXTA'] = df.DAY_OF_WEEK.apply(util.get_day_of_week_binary, args=(5,))
    df['SABADO'] = df.DAY_OF_WEEK.apply(util.get_day_of_week_binary, args=(6,))
    df['DOMINGO'] = df.DAY_OF_WEEK.apply(util.get_day_of_week_binary, args=(7,))

    # I will now determine when the user turns the TV off
    df['DATA_HEAD'] = df.groupby(['BOX_ID'])['ORIGIN_TIME'].shift(-1)

    df['DIFFERENCE'] = (pd.to_datetime(df["DATA_HEAD"], format="%Y-%m-%d %h:%mi:%s") -
                        pd.to_datetime(df["ORIGIN_TIME"], format="%Y-%m-%d %h:%mi:%s"))

    df['DIFFERENCE_CHECK'] = df['DIFFERENCE'].isnull()

    df['EFFECTIVE_DURATION'] = df.apply(util.diff, axis=1)

    df['TURN_OFF'] = df.apply(util.turn_off, axis=1)

    df['CHANNEL_HEAD'] = df.groupby(['BOX_ID'])['CHANNEL_NAME'].shift(-1)

    # Now I will determine when the user changes channel
    df['CHANNEL_CHANGE'] = df.apply(util.change_channel, axis=1)

    print 'Calculating features step two...'

    # Creates a rank of the MOST WATCHED by program / channel / CHANNEL_GENRE
    # 1. by program
    df = df.join(df.groupby(['PROGRAM_NAME'])['DURATION_MIN'].sum(), on='PROGRAM_NAME', rsuffix='_SUM_PROGRAM_NAME')
    df['RANK_PROGRAM_NAME'] = df['DURATION_MIN_SUM_PROGRAM_NAME'].rank(method='dense', ascending=False).to_frame()

    # 2. by channel
    df = df.join(df.groupby(['CHANNEL_NAME'])['DURATION_MIN'].sum(), on='CHANNEL_NAME', rsuffix='_SUM_CHANNEL_NAME')
    df['RANK_CHANNEL_NAME'] = df['DURATION_MIN_SUM_CHANNEL_NAME'].rank(method='dense', ascending=False).to_frame()

    # 3. by genre
    df = df.join(df.groupby(['CHANNEL_GENRE'])['DURATION_MIN'].sum(), on='CHANNEL_GENRE', rsuffix='_SUM_CHANNEL_GENRE')
    df['RANK_CHANNEL_GENRE'] = df['DURATION_MIN_SUM_CHANNEL_GENRE'].rank(method='dense', ascending=False).to_frame()

    # Creates a rank of the CHANNEL_CHANGE  by PROGRAM_NAME / CHANNEL_NAME / CHANNEL_GENRE
    # 1. by program
    df = df.join(df.groupby(['PROGRAM_NAME'])['CHANNEL_CHANGE'].sum(), on='PROGRAM_NAME', rsuffix='_SUM_PROGRAM_NAME')
    df['RANK_PROGRAM_NAME_CHANNEL_CHANGE'] = df['CHANNEL_CHANGE_SUM_PROGRAM_NAME'].rank(method='dense', ascending=False).to_frame()

    # 2. by channel
    df = df.join(df.groupby(['CHANNEL_NAME'])['CHANNEL_CHANGE'].sum(), on='CHANNEL_NAME', rsuffix='_SUM_CHANNEL_NAME')
    df['RANK_CHANNEL_NAME_CHANNEL_CHANGE'] = df['CHANNEL_CHANGE_SUM_CHANNEL_NAME'].rank(method='dense', ascending=False).to_frame()

    # 3. by genre
    df = df.join(df.groupby(['CHANNEL_GENRE'])['CHANNEL_CHANGE'].sum(), on='CHANNEL_GENRE', rsuffix='_SUM_CHANNEL_GENRE')
    df['RANK_CHANNEL_GENRE_CHANNEL_CHANGE'] = df['CHANNEL_CHANGE_SUM_CHANNEL_GENRE'].rank(method='dense', ascending=False).to_frame()

    # Determines if the event is HD or not
    df['FLG_IS_HD'] = df.IS_HD.apply(util.get_flg_is_hd)

    # Determines the PROGRAM PER USER that has less change channel
    # Get the sum of change channel per user
    df = df.join(df.groupby(['BOX_ID', 'PROGRAM_NAME'])['CHANNEL_CHANGE'].sum(), on=['BOX_ID', 'PROGRAM_NAME'],
                 rsuffix='_PER_USER_TEMP_P')

    # Get the rank of the sum
    df['RANK_OBJ_PROGRAM'] = df.groupby(['BOX_ID'])['CHANNEL_CHANGE_PER_USER_TEMP_P'].rank(method='dense',
                                                                                           ascending=False).to_frame()

    # Determines the CHANNEL PER USER that has less change channel
    # Get the sum of change channel per user
    df = df.join(df.groupby(['BOX_ID', 'CHANNEL_NAME'])['CHANNEL_CHANGE'].sum(), on=['BOX_ID', 'CHANNEL_NAME'],
                 rsuffix='_PER_USER_TEMP_C')

    # Get the rank of the sum
    df['RANK_OBJ_CHANNEL'] = df.groupby(['BOX_ID'])['CHANNEL_CHANGE_PER_USER_TEMP_C'].rank(method='dense',
                                                                                           ascending=False).to_frame()

    print 'Calculating features step three...'

    # Calculating cluster variable in binary
    df['CLUSTER_KIDS'] = df.CLUSTER.apply(util.get_cluster_binary, args=(1,))
    df['CLUSTER_GENERAL'] = df.CLUSTER.apply(util.get_cluster_binary, args=(2,))
    df['CLUSTER_MUSIC'] = df.CLUSTER.apply(util.get_cluster_binary, args=(3,))
    df['CLUSTER_SPORTS'] = df.CLUSTER.apply(util.get_cluster_binary, args=(4,))

    # Calculating program variable in binary
    df['CATEGORY_COOKING'] = df.CATEGORY.apply(util.get_program_name_binary, args=(1,))
    df['CATEGORY_SPORTS'] = df.CATEGORY.apply(util.get_program_name_binary, args=(2,))
    df['CATEGORY_DOCUMENTARIES'] = df.CATEGORY.apply(util.get_program_name_binary, args=(3,))
    df['CATEGORY_MOVIES'] = df.CATEGORY.apply(util.get_program_name_binary, args=(4,))
    df['CATEGORY_KIDS'] = df.CATEGORY.apply(util.get_program_name_binary, args=(5,))
    df['CATEGORY_INFORMATION'] = df.CATEGORY.apply(util.get_program_name_binary, args=(6,))
    df['CATEGORY_MUSIC'] = df.CATEGORY.apply(util.get_program_name_binary, args=(7,))
    df['CATEGORY_SOAP_OPERAS'] = df.CATEGORY.apply(util.get_program_name_binary, args=(8,))
    df['CATEGORY_SERIES'] = df.CATEGORY.apply(util.get_program_name_binary, args=(9,))
    df['CATEGORY_CONTESTS'] = df.CATEGORY.apply(util.get_program_name_binary, args=(10,))
    df['CATEGORY_ADULTS'] = df.CATEGORY.apply(util.get_program_name_binary, args=(11,))
    df['CATEGORY_OTHERS'] = df.CATEGORY.apply(util.get_program_name_binary, args=(12,))

    print 'Fixing missing values...'

    df["RANK_CHANNEL_NAME"].fillna(df["RANK_CHANNEL_NAME"].median(), inplace=True)
    df["RANK_PROGRAM_NAME"].fillna(df["RANK_PROGRAM_NAME"].median(), inplace=True)
    df["RANK_CHANNEL_GENRE"].fillna(df["RANK_CHANNEL_GENRE"].median(), inplace=True)
    df["RANK_PROGRAM_NAME_CHANNEL_CHANGE"].fillna(df["RANK_PROGRAM_NAME_CHANNEL_CHANGE"].median(), inplace=True)
    df["RANK_CHANNEL_NAME_CHANNEL_CHANGE"].fillna(df["RANK_CHANNEL_NAME_CHANNEL_CHANGE"].median(), inplace=True)
    df["RANK_CHANNEL_GENRE_CHANNEL_CHANGE"].fillna(df["RANK_CHANNEL_GENRE_CHANNEL_CHANGE"].median(), inplace=True)
    df["RANK_OBJ_CHANNEL"].fillna(df["RANK_OBJ_CHANNEL"].median(), inplace=True)
    df["RANK_OBJ_PROGRAM"].fillna(df["RANK_OBJ_PROGRAM"].median(), inplace=True)

    print 'Writing the table to the database...'

    util.write_table_to_my_sql(df, 't_python_events', 'ALTER TABLE T_PYTHON_EVENTS ADD PRIMARY KEY (ID)', 'local')

    print 'Success'
