# coding=utf-8

"""
.. module:: model_change_channel_2.py
    :synopsis: This module create clusters within the several clients
               It also performs PCA and prints an image of the given clusters.
               This module is done after variable selection trough anova tests in model_change_channel.py

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
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle

print 'Fetching the dataframe...'

df = util.get_df_from_conn('SELECT BOX_ID, TOTAL_MINUTES, C_GENERALIST_MINUTES,FRIDAY_MINUTES, '
                           'P_CONTESTS_MINUTES, P_COOKING_MINUTES, P_SPORTS_MINUTES,'
                           ' P_DOCUMENTARIES_MINUTES, P_MOVIES_MINUTES, P_KIDS_MINUTES, P_INFORMATION_MINUTES, '
                           'P_MUSIC_MINUTES, P_SOAP_OPERAS_MINUTES, P_OTHERS_MINUTES, P_SERIES_MINUTES'
                           ' FROM T_SEGMENT_USERS_DATA_NORMALIZED', 'local')

df_use = df.ix[:, 1:]

df_norm = (df_use - df_use.mean()) / (df_use.max() - df_use.min())

df_use = scale(df_use)

sample_size = df.BOX_ID.count()

n_clusters = 4

print 'Performing Kmeans model...'

kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
kmeans.fit(df_use)

df_norm['CLUSTER'] = kmeans.predict(df_use)

print 'Printing metrics...'

print metrics.silhouette_score(df_use, kmeans.labels_, metric='euclidean', sample_size=sample_size)

# The score is bounded between -1 for incorrect clustering and +1 for highly dense clustering.
# Scores around zero indicate overlapping clusters.
# The score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster.

print 'Performing PCA model...'

reduced_data = PCA(n_components=2).fit_transform(df_use)
kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
kmeans.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

print 'Saving the model for later usage...'

with open('k_means_plus.pk', 'wb') as fin:
    pickle.dump(kmeans, fin)

print 'Writing the table to the database...'

df_norm['BOX_ID'] = df['BOX_ID']

util.write_table_to_my_sql(df_norm, 't_segment_users_data_clusters_2', '', 'local')

print 'Success'
