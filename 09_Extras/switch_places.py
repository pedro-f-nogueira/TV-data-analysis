# coding=utf-8

"""
.. module:: switch_places.py
    :synopsis: This module transfers tables from one schema to another

.. moduleauthor:: Pedro Nogueira <pedro.fig.nogueira@gmail.com>
"""

import os
import sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)
del path

import implicit

import utilities.utilities as util

print 'Fetching records...'

df = util.get_df_from_conn('SELECT * FROM T_PROG_RECOMMENDATION_TOTAL;', 'local')

print 'Writing the table to the database...'

util.write_table_to_my_sql(df, 'T_PROG_RECOMMENDATION_TOTAL', '', 'remote')

print 'Success...'