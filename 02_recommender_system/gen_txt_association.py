# coding=utf-8

"""
.. module:: model_user_rec.py
    :synopsis: This module will give recommendations for programs based on user info

.. moduleauthor:: Pedro Nogueira <pedro.fig.nogueira@gmail.com>
"""

import sqlalchemy as sql
import pandas as pd


def get_df_from_conn(i_query, i_place):
    """
    This function creates a connection to the database
    Returns a df according to the input query

    i_query -> query to be returned to the dataframe
    """

    if i_place == 'aws':
        config = ''
    elif i_place == 'remote':
        config = 'mysql://pedro.fig.nogueira:changeme@94.76.213.231:3306/pgbia_altice_v2?charset=utf8&use_unicode=True'
    else:
        config = 'mysql://root:girafa@127.0.0.1:3306/altice?charset=utf8&use_unicode=True'

    engine = sql.create_engine(
        config,
        pool_size=100,
        pool_recycle=3600,
        )
    db = engine.connect()
    my_df = pd.read_sql(i_query, con=db)
    db.close()

    return my_df


print 'Fetching records...'

df = get_df_from_conn('SELECT BOX_ID_WEEK, PROGRAM_NAME FROM T_PROG_ASSOCIATION ORDER BY BOX_ID_WEEK', 'local')

a = 'nothing at all'
save = ''
total = ''

print 'ola'

myfile = open('association.txt', 'w')

for index, row in df.iterrows():
    if a == 'nothing at all':
        a == row[0].encode('utf-8')

    if row[0] == a:
        save = save + row[1].encode('utf-8') + ' *** '
    else:
        total = save + '\n'
        myfile.write(total)
        save = ''

    a = row[0]


myfile.close()

