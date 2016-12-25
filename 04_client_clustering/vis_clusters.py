# coding=utf-8

"""
.. module:: vis_clusters.py
    :synopsis: This module creates graphs in order to better interpret the clusters created.

.. moduleauthor:: Pedro Nogueira <pedro.fig.nogueira@gmail.com>
"""

import os
import sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)
del path

import utilities.utilities as util
from pandas.tools.plotting import parallel_coordinates
import matplotlib.pyplot as plt
plt.style.use('ggplot')

print 'Fetching the dataframe...'

df = util.get_df_from_conn('SELECT CASE WHEN CLUSTER = 0 THEN \'Kids\' WHEN CLUSTER = 1 THEN \'Music\''
                           'WHEN CLUSTER = 2 THEN \'Sports\' ELSE \'General\' END AS CLUSTER, '
                           'AVG(TOTAL_MINUTES) AS Total, AVG(C_GENERALIST_MINUTES) as C_Generalist,'
                           'AVG(FRIDAY_MINUTES) as Friday, AVG(P_CONTESTS_MINUTES) as P_Contests, '
                           'AVG(P_COOKING_MINUTES) as P_Cooking, AVG(P_SPORTS_MINUTES) as P_Sports, '
                           'AVG(P_DOCUMENTARIES_MINUTES) as P_Document, AVG(P_MOVIES_MINUTES) as P_Movies, '
                           'AVG(P_KIDS_MINUTES) as P_Kids, AVG(P_INFORMATION_MINUTES) as P_Inform, '
                           'AVG(P_MUSIC_MINUTES) as P_Music, AVG(P_SOAP_OPERAS_MINUTES) as P_Soap, '
                           'AVG(P_OTHERS_MINUTES) as P_Others, AVG(P_SERIES_MINUTES) P_Series'
                           ' FROM T_SEGMENT_USERS_DATA_CLUSTERS_2 GROUP BY CLUSTER', 'local')

print 'Printing graphics...'

data = df

plt.figure(num=None, figsize=(30, 12), dpi=80)
parallel_coordinates(data, 'CLUSTER', linewidth=8.0)

plt.title('Average of each variable for its given Cluster')

plt.show()

print 'Success'

