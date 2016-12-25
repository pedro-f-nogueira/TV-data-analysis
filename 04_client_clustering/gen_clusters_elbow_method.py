# coding=utf-8

"""
.. module:: gen_clusters_elbow_method_2.py
    :synopsis: This module determines the best number of clusters in the clients dataset.
               This module is done after variable selection trough anova tests in gen_clusters_elbow_method.py

.. moduleauthor:: Pedro Nogueira <pedro.fig.nogueira@gmail.com>
"""
import os
import sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)
del path

import utilities.utilities as util
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

print 'Fetching the dataframe...'

df = util.get_df_from_conn('SELECT BOX_ID, TOTAL_MINUTES, C_GENERALIST_MINUTES,FRIDAY_MINUTES, '
                           'P_CONTESTS_MINUTES, P_COOKING_MINUTES, P_SPORTS_MINUTES,'
                           ' P_DOCUMENTARIES_MINUTES, P_MOVIES_MINUTES, P_KIDS_MINUTES, P_INFORMATION_MINUTES, '
                           'P_MUSIC_MINUTES, P_SOAP_OPERAS_MINUTES, P_OTHERS_MINUTES, P_SERIES_MINUTES'
                           ' FROM T_SEGMENT_USERS_DATA_NORMALIZED', 'local')
df_use = df.ix[:, 1:]

df_norm = (df_use - df_use.mean()) / (df_use.max() - df_use.min())

df_use = scale(df_norm)

print 'Getting elbow method diagram...'

clusters=range(1,10)
meandist=[]

for k in clusters:
    model = KMeans(init='k-means++', n_clusters=k, n_init=10)
    model.fit(df_use)
    # Adding average euclidean distance between the model.cluster_centers_, the cluster centroids from the model
    # results, to array.
    meandist.append(sum(np.min(cdist(df_use, model.cluster_centers_, 'euclidean'), axis=1))
    / df_use.shape[0])


plt.plot(clusters, meandist)
plt.xlabel('Number of clusters')
plt.ylabel('Average distance')
plt.title('Selecting k with the Elbow Method')
plt.show()

print 'Writing Pearson correlation table to the database...'

print util.get_pearson_correlation(df)

"""
High correlations
P_MOVIES_MINUTES         P_SERIES_MINUTES           0.726455
P_SERIES_MINUTES         P_MOVIES_MINUTES           0.726455

P_INFORMATION_MINUTES    C_INFORM_MINUTES           0.869423
P_MUSIC_MINUTES          C_MUSIC_MINUTES            0.944259
C_MUSIC_MINUTES          P_MUSIC_MINUTES            0.944259
C_SPORTS_MINUTES         P_SPORTS_MINUTES           0.969992
P_KIDS_MINUTES           C_CHILD_MINUTES            0.998223
"""

print 'Success...'
