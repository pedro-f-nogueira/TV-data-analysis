# coding=utf-8

"""
.. module:: utilities.py
    :synopsis: This module provides several functions to other modules.

.. moduleauthor:: Pedro Nogueira <pedro.fig.nogueira@gmail.com>
"""

import pandas as pd
import sqlalchemy as sql
import re
from nltk.corpus import stopwords
from sklearn.cross_validation import train_test_split
from datetime import datetime
import numpy as np
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn import cross_validation
from sklearn import metrics
# from langdetect import detect


def get_df_from_conn(i_query, i_place):
    """
    This function creates a connection to the database
    Returns a df according to the input query

    i_query -> query to be returned to the dataframe
    """

    if i_place == 'aws':
        config = 'deleted'
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


def write_table_to_my_sql(i_df, i_table_name, i_command, i_place):
    """
    This function writes the df created to the database
    It will also execute a command according to the input

    i_df -> Input dataframe that will be written to the database
    i_table_name -> Table that will contain that dataframe
    i_command -> SQL command to be executed
    """

    if i_place == 'aws':
        config = 'deleted'
    elif i_place == 'remote':
        config = 'deleted'
    else:
        config = 'deleted'

    engine = sql.create_engine(
        config,
        pool_size=100,
        pool_recycle=3600)
    db = engine.connect()

    db.execute('DROP TABLE IF EXISTS ' + i_table_name)

    i_df.to_sql(i_table_name,
                con=engine,
                if_exists='replace',
                chunksize=1000,
                index=False)

    if i_command != '':
        db.execute(i_command)

    return


def review_words(i_text):
    """
    Input -> Text to be cleaned
    Output -> Clean unicode text
    """
    # 1. Remove non-letters
    letters_only = re.sub("[.,“”/#!?$%\^&\*;:{}=\-_`~()']",  # The pattern to search for
                          " ",  # The pattern to replace it with
                          i_text)  # The text to search

    # 2. Convert to lower case, split into individual words
    words = letters_only.lower().split()

    # 3. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("portuguese"))

    # 4. Add stop words defined by me
    a = [u'ser', u'é', u'vai', u'têm', u'vão', u'sobre', u'sobre', u'cada', u'onde', u'tudo']

    for i in a:
        stops.add(i)

    # 5. Remove stop words
    meaningful_words = [w for w in words if w not in stops]

    # 6. Join the words back into one string separated by space,
    # and return the result.
    meaningful_words = " ".join(meaningful_words)

    # 7. Removes expressions defined by me
    meaningful_words = meaningful_words.replace(u'classificação etária m 16', '')
    meaningful_words = meaningful_words.replace(u'classificação etária m 12', '')
    meaningful_words = meaningful_words.replace(u'classificação etária m 6', '')
    meaningful_words = meaningful_words.replace(u'class etária m 6', '')
    meaningful_words = meaningful_words.replace(u'class etária m 12', '')
    meaningful_words = meaningful_words.replace(u'class etária m 16', '')

    # 8. returns clean text
    return meaningful_words


def get_train_test_target(x, y):
    """
    Divides train and test randomly with 1/3 going to test and 2/3 going to train
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42, stratify=y)

    return x_train, x_test, y_train, y_test


def split_train_test_val_det(i_date):
    """
    Divides train, test and validation datasets acordding to a time variable
    """
    if i_date < datetime.strptime('2016-07-16 00:00:00', '%Y-%m-%d %H:%M:%S'):
        return 1
    elif datetime.strptime('2016-07-16 00:00:00', '%Y-%m-%d %H:%M:%S') \
            < i_date < datetime.strptime('2016-08-16 00:00:00', '%Y-%m-%d %H:%M:%S'):
        return 2
    else:
        return 3


def get_channel_genre(i_genre, *args):
    """
    Generates binary variables using the categorical genre variable
    """
    i_genre = i_genre.encode('utf-8')

    if i_genre == 'Generalistas' and args[0] == 1:
        return 1
    elif i_genre == 'Informação' and args[0] == 2:
        return 1
    elif i_genre == 'Entretenimento' and args[0] == 3:
        return 1
    elif i_genre == 'Desporto' and args[0] == 4:
        return 1
    elif i_genre == 'Nacionais' and args[0] == 5:
        return 1
    elif i_genre == 'Infantil' and args[0] == 6:
        return 1
    elif i_genre == 'Outros' and args[0] == 7:
        return 1
    elif i_genre == 'Séries' and args[0] == 8:
        return 1
    elif i_genre == 'Lifestyle' and args[0] == 9:
        return 1
    elif i_genre == 'Documentários' and args[0] == 10:
        return 1
    elif i_genre == 'Internacionais' and args[0] == 11:
        return 1
    elif i_genre == 'Adultos' and args[0] == 12:
        return 1
    else:
        return 0


def get_period_of_day(i_hour):
    """
    Generates period of day variable using the the hours numerical variable.
    """
    if 0 < i_hour < 7:
        return 'night'
    elif 7 <= i_hour < 12:
        return 'morning'
    elif 12 <= i_hour < 20:
        return 'afternoon'
    else:
        return 'evening'


def get_period_of_day_binary(i_text, *args):
    """
    Generates binary variables using the categorical period of day variable
    """
    if i_text == 'morning' and args[0] == 1:
        return '1'
    elif i_text == 'afternoon' and args[0] == 2:
        return '1'
    elif i_text == 'evening' and args[0] == 3:
        return '1'
    elif i_text == 'night' and args[0] == 4:
        return '1'
    else:
        return 0


def get_day_of_week(i_date):
    """
   Generates day of week variable
   """
    day_of_week = i_date.strftime('%A')

    if day_of_week == 'Monday':
        return 'Segunda'
    elif day_of_week == 'Tuesday':
        return 'Terca'
    elif day_of_week == 'Wednesday':
        return 'Quarta'
    elif day_of_week == 'Thursday':
        return 'Quinta'
    elif day_of_week == 'Friday':
        return 'Sexta'
    elif day_of_week == 'Saturday':
        return 'Sabado'
    elif day_of_week == 'Sunday':
        return 'Domingo'
    else:
        return 'Erro'


def get_day_of_week_binary(i_text, *args):
    """
    Generates binary variables using the categorical day of week variable
    """
    if i_text == 'Segunda' and args[0] == 1:
        return '1'
    elif i_text == 'Terca' and args[0] == 2:
        return '1'
    elif i_text == 'Quarta' and args[0] == 3:
        return '1'
    elif i_text == 'Quinta' and args[0] == 4:
        return '1'
    elif i_text == 'Sexta' and args[0] == 5:
        return '1'
    elif i_text == 'Sabado' and args[0] == 6:
        return '1'
    elif i_text == 'Domingo' and args[0] == 7:
        return '1'
    else:
        return 0


def gen_rand_train_test(i_len):
    return np.random.choice([1, 0], i_len, p=[0.7, 0.3])


def stop_watch(row):
    if row['RANK'] == row['RANK_MAX']:
        return 1
    else:
        return 0


def turn_off(row):
    if row['EFFECTIVE_DURATION'] > row['DURATION_MIN'] + 10:
        return 1
    else:
        return 0


def diff(row):
    if row['DIFFERENCE_CHECK']:
        return 0
    else:
        return round((row['DIFFERENCE'].total_seconds())/60)


def change_channel(row):
    if row['CHANNEL_HEAD'] == row['CHANNEL_NAME'] or row['TURN_OFF'] == 1:
        return 0
    else:
        return 1


def get_genre_per_id(i_df_box, i_df):
    """
    Determines the sum of minutes watched grouped by user PER GENRE
    """
    # 1.Informação
    i_df_box = i_df_box.join(i_df[(i_df['CHANNEL_GENRE'] == 'Informação'.decode('utf-8'))].groupby(['BOX_ID'])['DURATION_MIN'].sum(),
                             on='BOX_ID')
    i_df_box.rename(columns={'DURATION_MIN': 'C_INFORM_MINUTES'}, inplace=True)

    # 2.Lifestyle
    i_df_box = i_df_box.join(i_df[(i_df['CHANNEL_GENRE'] == 'Lifestyle')].groupby(['BOX_ID'])['DURATION_MIN'].sum(),
                             on='BOX_ID')
    i_df_box.rename(columns={'DURATION_MIN': 'C_LIFESTYLE_MINUTES'}, inplace=True)

    # 3.Infantil
    i_df_box = i_df_box.join(i_df[(i_df['CHANNEL_GENRE'] == 'Infantil')].groupby(['BOX_ID'])['DURATION_MIN'].sum(),
                             on='BOX_ID')
    i_df_box.rename(columns={'DURATION_MIN': 'C_CHILD_MINUTES'}, inplace=True)

    # 4.Filmes
    i_df_box = i_df_box.join(i_df[(i_df['CHANNEL_GENRE'] == 'Filmes')].groupby(['BOX_ID'])['DURATION_MIN'].sum(),
                             on='BOX_ID')
    i_df_box.rename(columns={'DURATION_MIN': 'C_MOVIES_MINUTES'}, inplace=True)

    # 5.Entretenimento
    i_df_box = i_df_box.join(
        i_df[(i_df['CHANNEL_GENRE'] == 'Entretenimento')].groupby(['BOX_ID'])['DURATION_MIN'].sum(),
        on='BOX_ID')
    i_df_box.rename(columns={'DURATION_MIN': 'C_ENTERTAIN_MINUTES'}, inplace=True)

    # 6.Desporto
    i_df_box = i_df_box.join(i_df[(i_df['CHANNEL_GENRE'] == 'Desporto')].groupby(['BOX_ID'])['DURATION_MIN'].sum(),
                             on='BOX_ID')
    i_df_box.rename(columns={'DURATION_MIN': 'C_SPORTS_MINUTES'}, inplace=True)

    # 7.Generalistas
    i_df_box = i_df_box.join(i_df[(i_df['CHANNEL_GENRE'] == 'Generalistas')].groupby(['BOX_ID'])['DURATION_MIN'].sum(),
                             on='BOX_ID')
    i_df_box.rename(columns={'DURATION_MIN': 'C_GENERALIST_MINUTES'}, inplace=True)

    # 8.Música
    i_df_box = i_df_box.join(i_df[(i_df['CHANNEL_GENRE'] == 'Música'.decode('utf-8'))].groupby(['BOX_ID'])['DURATION_MIN'].sum(),
                             on='BOX_ID')
    i_df_box.rename(columns={'DURATION_MIN': 'C_MUSIC_MINUTES'}, inplace=True)

    # 9.Séries
    i_df_box = i_df_box.join(i_df[(i_df['CHANNEL_GENRE'] == 'Séries'.decode('utf-8'))].groupby(['BOX_ID'])['DURATION_MIN'].sum(),
                             on='BOX_ID')
    i_df_box.rename(columns={'DURATION_MIN': 'C_SERIES_MINUTES'}, inplace=True)

    # 10.Nacionais
    i_df_box = i_df_box.join(i_df[(i_df['CHANNEL_GENRE'] == 'Nacionais')].groupby(['BOX_ID'])['DURATION_MIN'].sum(),
                             on='BOX_ID')
    i_df_box.rename(columns={'DURATION_MIN': 'C_NATIONAL_MINUTES'}, inplace=True)

    # 11.Documentários
    i_df_box = i_df_box.join(i_df[(i_df['CHANNEL_GENRE'] == 'Documentários'.decode('utf-8'))].groupby(['BOX_ID'])['DURATION_MIN'].sum(),
                             on='BOX_ID')
    i_df_box.rename(columns={'DURATION_MIN': 'C_DOCUMENTARIES_MINUTES'}, inplace=True)

    # 12.Internacionais
    i_df_box = i_df_box.join(
        i_df[(i_df['CHANNEL_GENRE'] == 'Internacionais')].groupby(['BOX_ID'])['DURATION_MIN'].sum(),
        on='BOX_ID')
    i_df_box.rename(columns={'DURATION_MIN': 'C_INTERNATIONAL_MINUTES'}, inplace=True)

    # 13.Outros
    i_df_box = i_df_box.join(i_df[(i_df['CHANNEL_GENRE'] == 'Outros')].groupby(['BOX_ID'])['DURATION_MIN'].sum(),
                             on='BOX_ID')
    i_df_box.rename(columns={'DURATION_MIN': 'C_OTHERS_MINUTES'}, inplace=True)

    # 14.Adultos
    i_df_box = i_df_box.join(i_df[(i_df['CHANNEL_GENRE'] == 'Adultos')].groupby(['BOX_ID'])['DURATION_MIN'].sum(),
                             on='BOX_ID')
    i_df_box.rename(columns={'DURATION_MIN': 'C_ADULTS_MINUTES'}, inplace=True)

    # 15.undefined
    i_df_box = i_df_box.join(i_df[(i_df['CHANNEL_GENRE'] == 'undefined')].groupby(['BOX_ID'])['DURATION_MIN'].sum(),
                             on='BOX_ID')
    i_df_box.rename(columns={'DURATION_MIN': 'C_UNDEFINED_MINUTES'}, inplace=True)

    i_df_box["C_INFORM_MINUTES"].fillna(value=0, inplace=True)
    i_df_box["C_LIFESTYLE_MINUTES"].fillna(value=0, inplace=True)
    i_df_box["C_CHILD_MINUTES"].fillna(value=0, inplace=True)
    i_df_box["C_MOVIES_MINUTES"].fillna(value=0, inplace=True)
    i_df_box["C_ENTERTAIN_MINUTES"].fillna(value=0, inplace=True)
    i_df_box["C_SPORTS_MINUTES"].fillna(value=0, inplace=True)
    i_df_box["C_GENERALIST_MINUTES"].fillna(value=0, inplace=True)
    i_df_box["C_MUSIC_MINUTES"].fillna(value=0, inplace=True)
    i_df_box["C_SERIES_MINUTES"].fillna(value=0, inplace=True)
    i_df_box["C_NATIONAL_MINUTES"].fillna(value=0, inplace=True)
    i_df_box["C_DOCUMENTARIES_MINUTES"].fillna(value=0, inplace=True)
    i_df_box["C_INTERNATIONAL_MINUTES"].fillna(value=0, inplace=True)
    i_df_box["C_OTHERS_MINUTES"].fillna(value=0, inplace=True)
    i_df_box["C_ADULTS_MINUTES"].fillna(value=0, inplace=True)
    i_df_box["C_UNDEFINED_MINUTES"].fillna(value=0, inplace=True)

    return i_df_box


def get_pod_per_id(i_df_box, i_df):
    """
    Determines the sum of minutes watched grouped by user PER PERIOD OF DAY
    """
    # 1.evening
    i_df_box = i_df_box.join(
        i_df[(i_df['PERIOD_OF_DAY'] == 'evening'.decode('utf-8'))].groupby(['BOX_ID'])['DURATION_MIN'].sum(),
        on='BOX_ID')
    i_df_box.rename(columns={'DURATION_MIN': 'EVENING_MINUTES'}, inplace=True)

    # 2.afternoon
    i_df_box = i_df_box.join(
        i_df[(i_df['PERIOD_OF_DAY'] == 'afternoon'.decode('utf-8'))].groupby(['BOX_ID'])['DURATION_MIN'].sum(),
        on='BOX_ID')
    i_df_box.rename(columns={'DURATION_MIN': 'AFTERNOON_MINUTES'}, inplace=True)

    # 3.night
    i_df_box = i_df_box.join(
        i_df[(i_df['PERIOD_OF_DAY'] == 'night'.decode('utf-8'))].groupby(['BOX_ID'])['DURATION_MIN'].sum(),
        on='BOX_ID')
    i_df_box.rename(columns={'DURATION_MIN': 'NIGHT_MINUTES'}, inplace=True)

    # 4.morning
    i_df_box = i_df_box.join(
        i_df[(i_df['PERIOD_OF_DAY'] == 'morning'.decode('utf-8'))].groupby(['BOX_ID'])['DURATION_MIN'].sum(),
        on='BOX_ID')
    i_df_box.rename(columns={'DURATION_MIN': 'MORNING_MINUTES'}, inplace=True)

    i_df_box["EVENING_MINUTES"].fillna(value=0, inplace=True)
    i_df_box["AFTERNOON_MINUTES"].fillna(value=0, inplace=True)
    i_df_box["NIGHT_MINUTES"].fillna(value=0, inplace=True)
    i_df_box["MORNING_MINUTES"].fillna(value=0, inplace=True)

    return i_df_box


def get_dow_per_id(i_df_box, i_df):
    """
    Determines the sum of minutes watched grouped by user PER DAY OF WEEK
    """
    # 1.Segunda
    i_df_box = i_df_box.join(
        i_df[(i_df['DAY_OF_WEEK'] == 'Segunda'.decode('utf-8'))].groupby(['BOX_ID'])['DURATION_MIN'].sum(),
        on='BOX_ID')
    i_df_box.rename(columns={'DURATION_MIN': 'MONDAY_MINUTES'}, inplace=True)

    # 2.Terca
    i_df_box = i_df_box.join(
        i_df[(i_df['DAY_OF_WEEK'] == 'Terca'.decode('utf-8'))].groupby(['BOX_ID'])['DURATION_MIN'].sum(),
        on='BOX_ID')
    i_df_box.rename(columns={'DURATION_MIN': 'TUESDAY_MINUTES'}, inplace=True)

    # 3.Quarta
    i_df_box = i_df_box.join(
        i_df[(i_df['DAY_OF_WEEK'] == 'Quarta'.decode('utf-8'))].groupby(['BOX_ID'])['DURATION_MIN'].sum(),
        on='BOX_ID')
    i_df_box.rename(columns={'DURATION_MIN': 'WEDNESDAY_MINUTES'}, inplace=True)

    # 4.Quinta
    i_df_box = i_df_box.join(
        i_df[(i_df['DAY_OF_WEEK'] == 'Quinta'.decode('utf-8'))].groupby(['BOX_ID'])['DURATION_MIN'].sum(),
        on='BOX_ID')
    i_df_box.rename(columns={'DURATION_MIN': 'THURSDAY_MINUTES'}, inplace=True)

    # 5.Sexta
    i_df_box = i_df_box.join(
        i_df[(i_df['DAY_OF_WEEK'] == 'Sexta'.decode('utf-8'))].groupby(['BOX_ID'])['DURATION_MIN'].sum(),
        on='BOX_ID')
    i_df_box.rename(columns={'DURATION_MIN': 'FRIDAY_MINUTES'}, inplace=True)

    # 6.Sabado
    i_df_box = i_df_box.join(
        i_df[(i_df['DAY_OF_WEEK'] == 'Sabado'.decode('utf-8'))].groupby(['BOX_ID'])['DURATION_MIN'].sum(),
        on='BOX_ID')
    i_df_box.rename(columns={'DURATION_MIN': 'SATURDAY_MINUTES'}, inplace=True)

    # 7.Domingo
    i_df_box = i_df_box.join(
        i_df[(i_df['DAY_OF_WEEK'] == 'Domingo'.decode('utf-8'))].groupby(['BOX_ID'])['DURATION_MIN'].sum(),
        on='BOX_ID')
    i_df_box.rename(columns={'DURATION_MIN': 'SUNDAY_MINUTES'}, inplace=True)

    i_df_box["MONDAY_MINUTES"].fillna(value=0, inplace=True)
    i_df_box["TUESDAY_MINUTES"].fillna(value=0, inplace=True)
    i_df_box["WEDNESDAY_MINUTES"].fillna(value=0, inplace=True)
    i_df_box["THURSDAY_MINUTES"].fillna(value=0, inplace=True)
    i_df_box["FRIDAY_MINUTES"].fillna(value=0, inplace=True)
    i_df_box["SATURDAY_MINUTES"].fillna(value=0, inplace=True)
    i_df_box["SUNDAY_MINUTES"].fillna(value=0, inplace=True)

    return i_df_box


def get_total_min_per_id(i_df_box, i_df):
    """
    Determines the sum of minutes watched grouped by user
    """
    i_df_box = i_df_box.join(i_df.groupby(['BOX_ID'])['DURATION_MIN'].sum(), on='BOX_ID')

    i_df_box.rename(columns={'DURATION_MIN': 'TOTAL_MINUTES'}, inplace=True)

    return i_df_box


def get_cat_prog_per_id(i_df_box, i_df):
    """
    Determines the sum of minutes watched grouped by user PER PROGRAM CATEGORY
    """
    # 1.ADULTS
    i_df_box = i_df_box.join(
        i_df[(i_df['CATEGORY'] == 'ADULTOS')].groupby(['BOX_ID'])['DURATION_MIN'].sum(),
        on='BOX_ID')
    i_df_box.rename(columns={'DURATION_MIN': 'P_ADULT_MINUTES'}, inplace=True)

    # 2.CONTESTS
    i_df_box = i_df_box.join(
        i_df[(i_df['CATEGORY'] == 'CONCURSOS')].groupby(['BOX_ID'])['DURATION_MIN'].sum(),
        on='BOX_ID')
    i_df_box.rename(columns={'DURATION_MIN': 'P_CONTESTS_MINUTES'}, inplace=True)

    # 3.COOKING
    i_df_box = i_df_box.join(
        i_df[(i_df['CATEGORY'] == 'CULINARIA')].groupby(['BOX_ID'])['DURATION_MIN'].sum(),
        on='BOX_ID')
    i_df_box.rename(columns={'DURATION_MIN': 'P_COOKING_MINUTES'}, inplace=True)

    # 4.SPORTS
    i_df_box = i_df_box.join(
        i_df[(i_df['CATEGORY'] == 'DESPORTO')].groupby(['BOX_ID'])['DURATION_MIN'].sum(),
        on='BOX_ID')
    i_df_box.rename(columns={'DURATION_MIN': 'P_SPORTS_MINUTES'}, inplace=True)

    # 5.DOCUMENTARIES
    i_df_box = i_df_box.join(
        i_df[(i_df['CATEGORY'] == 'DOCUMENTARIOS')].groupby(['BOX_ID'])['DURATION_MIN'].sum(),
        on='BOX_ID')
    i_df_box.rename(columns={'DURATION_MIN': 'P_DOCUMENTARIES_MINUTES'}, inplace=True)

    # 6.MOVIES
    i_df_box = i_df_box.join(
        i_df[(i_df['CATEGORY'] == 'FILMES')].groupby(['BOX_ID'])['DURATION_MIN'].sum(),
        on='BOX_ID')
    i_df_box.rename(columns={'DURATION_MIN': 'P_MOVIES_MINUTES'}, inplace=True)

    # 7.KIDS
    i_df_box = i_df_box.join(
        i_df[(i_df['CATEGORY'] == 'INFANTIL')].groupby(['BOX_ID'])['DURATION_MIN'].sum(),
        on='BOX_ID')
    i_df_box.rename(columns={'DURATION_MIN': 'P_KIDS_MINUTES'}, inplace=True)

    # 8.INFORMATION
    i_df_box = i_df_box.join(
        i_df[(i_df['CATEGORY'] == 'INFORMACAO')].groupby(['BOX_ID'])['DURATION_MIN'].sum(),
        on='BOX_ID')
    i_df_box.rename(columns={'DURATION_MIN': 'P_INFORMATION_MINUTES'}, inplace=True)

    # 9.MUSIC AND CULTURE
    i_df_box = i_df_box.join(
        i_df[(i_df['CATEGORY'] == 'MUSICA E CULTURA')].groupby(['BOX_ID'])['DURATION_MIN'].sum(),
        on='BOX_ID')
    i_df_box.rename(columns={'DURATION_MIN': 'P_MUSIC_MINUTES'}, inplace=True)

    # 10.SOAP_OPERAS
    i_df_box = i_df_box.join(
        i_df[(i_df['CATEGORY'] == 'NOVELAS')].groupby(['BOX_ID'])['DURATION_MIN'].sum(),
        on='BOX_ID')
    i_df_box.rename(columns={'DURATION_MIN': 'P_SOAP_OPERAS_MINUTES'}, inplace=True)

    # 11.OTHERS
    i_df_box = i_df_box.join(
        i_df[(i_df['CATEGORY'] == 'OUTROS')].groupby(['BOX_ID'])['DURATION_MIN'].sum(),
        on='BOX_ID')
    i_df_box.rename(columns={'DURATION_MIN': 'P_OTHERS_MINUTES'}, inplace=True)

    # 12.SERIES
    i_df_box = i_df_box.join(
        i_df[(i_df['CATEGORY'] == 'SERIES')].groupby(['BOX_ID'])['DURATION_MIN'].sum(),
        on='BOX_ID')
    i_df_box.rename(columns={'DURATION_MIN': 'P_SERIES_MINUTES'}, inplace=True)

    i_df_box["P_ADULT_MINUTES"].fillna(value=0, inplace=True)
    i_df_box["P_CONTESTS_MINUTES"].fillna(value=0, inplace=True)
    i_df_box["P_COOKING_MINUTES"].fillna(value=0, inplace=True)
    i_df_box["P_SPORTS_MINUTES"].fillna(value=0, inplace=True)
    i_df_box["P_DOCUMENTARIES_MINUTES"].fillna(value=0, inplace=True)
    i_df_box["P_MOVIES_MINUTES"].fillna(value=0, inplace=True)
    i_df_box["P_KIDS_MINUTES"].fillna(value=0, inplace=True)
    i_df_box["P_INFORMATION_MINUTES"].fillna(value=0, inplace=True)
    i_df_box["P_MUSIC_MINUTES"].fillna(value=0, inplace=True)
    i_df_box["P_SOAP_OPERAS_MINUTES"].fillna(value=0, inplace=True)
    i_df_box["P_OTHERS_MINUTES"].fillna(value=0, inplace=True)
    i_df_box["P_SERIES_MINUTES"].fillna(value=0, inplace=True)

    return i_df_box


def get_flg_is_hd(i_is_hd):
    """
    Determines is the event is HD or not
    """
    if i_is_hd == 'HD':
        return 1
    else:
        return 0


def perform_stemming(i_text):
    tokens = nltk.word_tokenize(i_text)
    porter = nltk.PorterStemmer()
    text_stem_porter = []
    for token in tokens:
        text_stem_porter.append(porter.stem(token))

    joint_stemmed = ','.join(text_stem_porter).strip(',')

    return joint_stemmed.replace(',', ' ')


def perform_lemmatization(i_text):
    tokens = nltk.word_tokenize(i_text)
    wnl = WordNetLemmatizer()
    text_lemma_wordnet = []
    for token in tokens:
        text_lemma_wordnet.append(wnl.lemmatize(token))

    joint_lemmatized = ','.join(text_lemma_wordnet).strip(',')

    return joint_lemmatized.replace(',', ' ')


def get_cluster_binary(i_text, *args):
    if i_text == 'Kids' and args[0] == 1:
        return '1'
    elif i_text == 'General' and args[0] == 2:
        return '1'
    elif i_text == 'Music' and args[0] == 3:
        return '1'
    elif i_text == 'Sports' and args[0] == 4:
        return '1'
    else:
        return 0


def get_program_name_binary(i_text, *args):
    if i_text == 'CULINARIA' and args[0] == 1:
        return '1'
    elif i_text == 'DESPORTO' and args[0] == 2:
        return '1'
    elif i_text == 'DOCUMENTARIOS' and args[0] == 3:
        return '1'
    elif i_text == 'FILMES' and args[0] == 4:
        return '1'
    elif i_text == 'INFANTIL' and args[0] == 5:
        return '1'
    elif i_text == 'INFORMACAO' and args[0] == 6:
        return '1'
    elif i_text == 'MUSICA E CULTURA' and args[0] == 7:
        return '1'
    elif i_text == 'NOVELAS' and args[0] == 8:
        return '1'
    elif i_text == 'SERIES' and args[0] == 9:
        return '1'
    elif i_text == 'CONCURSOS' and args[0] == 10:
        return '1'
    elif i_text == 'OUTROS' and args[0] == 11:
        return '1'
    elif i_text == 'Sports' and args[0] == 12:
        return '1'
    else:
        return 0


def get_pearson_correlation(i_df):
    """
    Calculates the pearson correlation in the dataframe and returns
    the sorted correlations
    """
    i_df = i_df.corr(method='pearson', min_periods=1)
    c = i_df.corr().abs()
    s = c.unstack()
    so = s.sort_values(kind="quicksort", ascending=False, na_position='last')
    so = so[so != 1]
    so = so[so > 0.8]

    return so


def evaluate_classifier_with_cv(clf, x_train, y_train, kfold, x_test, y_test):
    """
    Prints accuracy with cross validation
    Prints accuracy / precision / recall / f1 / confusion matrix without cross validation
    """

    # Train Results
    results = cross_validation.cross_val_score(clf, x_train, y_train, cv=kfold)
    print 'Train cross-validated accuracy: {}'
    print(results.mean())

    print('Train accuracy: {}'.format(clf.score(x_train, y_train)))

    y_predicted_train = clf.predict(x_train)
    print metrics.classification_report(y_train, y_predicted_train)

    print 'Printing confusion matrix...'

    matrix = metrics.confusion_matrix(y_train, y_predicted_train)
    print(matrix)

    # Test Results
    results = cross_validation.cross_val_score(clf, x_test, y_test, cv=kfold)
    print 'Test cross-validated accuracy: {}'
    print(results.mean())

    print('Test accuracy: {}'.format(clf.score(x_test, y_test)))

    y_predicted = clf.predict(x_test)
    print metrics.classification_report(y_test, y_predicted)

    print 'Printing confusion matrix...'

    matrix = metrics.confusion_matrix(y_test, y_predicted)
    print(matrix)


def evaluate_classifier(clf, x_train, y_train, x_test, y_test):
    """
    Prints accuracy / precision / recall / f1 / confusion matrix without cross validation
    """
    # Train Results
    print('Train accuracy: {}'.format(clf.score(x_train, y_train)))

    y_predicted_train = clf.predict(x_train)
    print metrics.classification_report(y_train, y_predicted_train)

    print 'Printing confusion matrix...'

    matrix = metrics.confusion_matrix(y_train, y_predicted_train)
    print(matrix)

    # Test Results
    print('Test accuracy: {}'.format(clf.score(x_test, y_test)))

    y_predicted = clf.predict(x_test)
    print metrics.classification_report(y_test, y_predicted)

    print 'Printing confusion matrix...'

    matrix = metrics.confusion_matrix(y_test, y_predicted)
    print(matrix)


"""
def detect_language(text_to_detect):
    # Returns the language of a given input text
    try:
        return detect(text_to_detect)
    except:
        return 'unknown'
"""
