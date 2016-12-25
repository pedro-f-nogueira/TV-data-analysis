# coding=utf-8

"""
.. module:: gen_t_python_programs.py
    :synopsis: This module generates cleans the program description and determines its the language

.. moduleauthor:: Pedro Nogueira <pedro.fig.nogueira@gmail.com>
"""
import os
import sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)
del path

import utilities.utilities as util
import re as re


def fix_season(self):
    """
    This function removes the t1, t2... from its input
    example
    input -> paramedics t1
    output -> paramedics
    """
    self = re.sub('\s+', ' ', self).strip()
    match = re.findall('.+?(?= t\d|Ep\d| - E p.)', self)

    if match:
        return ''.join(match).strip()
    else:
        return self.strip()


if __name__ == '__main__':
    print 'Fetching table...'

    df = util.get_df_from_conn('SELECT * FROM t_programs;')

    print 'Calculating features'

    # Removing t1, t2 from the name...
    df['NAME'] = df['NAME'].apply(fix_season)

    df['CLEAN_DESCR'] = df.DESCRIPTION.apply(util.review_words)

    print 'Detecting language...'
    # Will need to switch the project interpreter... this needs attention
    df['LANGUAGE'] = df.DESCRIPTION.apply(util.detect_language)

    print 'Writing the table to the database...'

    util.write_table_to_my_sql(df, 't_python_programs', 'ALTER TABLE T_PYTHON_PROGRAMS ADD PRIMARY KEY (ID)')

    print 'Success'
