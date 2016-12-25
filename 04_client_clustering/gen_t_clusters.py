# coding=utf-8

"""
.. module:: gen_t_python_events.py
    :synopsis: This module generates data to create clusters within the several clients

.. moduleauthor:: Pedro Nogueira <pedro.fig.nogueira@gmail.com>
"""

import os
import sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)
del path

import utilities.utilities as util

if __name__ == '__main__':
    print 'Fetching records...'

    df = util.get_df_from_conn('SELECT BOX_ID, DURATION_MIN, CHANNEL_GENRE, PERIOD_OF_DAY, DAY_OF_WEEK, CHANNEL_NAME, '
                               'TURN_OFF, CATEGORY FROM T_PYTHON_EVENTS;', 'local')

    print 'Calculating features...'

    df_use = df.drop_duplicates(['BOX_ID'])

    df_box = df_use['BOX_ID']

    df_box = df_box.to_frame(name='BOX_ID')

    # Calculate the sum of minutes watched PER user
    #0. TOTAL
    df_box = util.get_total_min_per_id(df_box, df)

    # 1. PER CHANNEL GENRE
    df_box = util.get_genre_per_id(df_box, df)

    # 2. PER PERIOD_OF_DAY
    df_box = util.get_pod_per_id(df_box, df)

    # 3. PER DAY OF WEEK
    df_box = util.get_dow_per_id(df_box, df)

    # 4. PER PROGRAM CATEGORY
    df_box = util.get_cat_prog_per_id(df_box, df)

    print 'Writing the table to the database...'

    util.write_table_to_my_sql(df_box, 't_segment_users_data', '', 'local')

    print 'Success'
