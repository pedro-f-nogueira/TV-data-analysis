# coding=utf-8


"""
.. module:: model_cluster_topics.py
    :synopsis: This module generates clusters using the programs description

.. moduleauthor:: Pedro Nogueira <pedro.fig.nogueira@gmail.com>
"""
from __future__ import print_function
import os
import sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)
del path

import utilities.utilities as util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import pandas as pd


def tokenize_only(i_text):
    return i_text.split(" ")

if __name__ == '__main__':

    df = util.get_df_from_conn('SELECT MASTER_CATEGORY, CLEAN_DESCR '
                               'FROM T_TRAINING_TEXT_MODEL '
                               'WHERE LENGTH(CLEAN_DESCR) > 10 '
                               , 'local')

    totalvocab_tokenized = []

    for i in df["CLEAN_DESCR"]:
        allwords_tokenized = tokenize_only(i)
        totalvocab_tokenized.extend(allwords_tokenized)

    vocab_frame = pd.DataFrame({'words': totalvocab_tokenized})

    print ('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')

    print ("Creating the Vectorizer...\n")

    # define vectorizer parameters
    tfidf_vectorizer = TfidfVectorizer(max_df=0.5, max_features=200000,
                                       min_df=0.005,
                                       use_idf=True,
                                       tokenizer=tokenize_only,
                                       ngram_range=(1, 3))

    vec = tfidf_vectorizer.fit(df['CLEAN_DESCR'])   # fit the vectorizer

    tfidf_matrix = vec.transform(df['CLEAN_DESCR'])  # fit the vectorizer to synopses

    print ("All done... Printing stuff...\n")

    terms = tfidf_vectorizer.get_feature_names()

    dist = 1 - cosine_similarity(tfidf_matrix)

    num_clusters = 12

    km = KMeans(n_clusters=num_clusters)

    km.fit(tfidf_matrix)

    clusters = km.labels_.tolist()

    joblib.dump(km,  'doc_cluster.pkl')

    km = joblib.load('doc_cluster.pkl')
    clusters = km.labels_.tolist()

    programs = {'MASTER_CATEGORY': df['MASTER_CATEGORY'].values.tolist(), 'descr': df['CLEAN_DESCR'].values.tolist(), 'cluster': clusters}

    frame = pd.DataFrame(programs, index=[clusters], columns=['MASTER_CATEGORY', 'descr', 'cluster'])

    print (frame['cluster'].value_counts())

    print("Top terms per cluster:")
    print()
    # sort cluster centers by proximity to centroid
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    for i in range(num_clusters):
        print("Cluster %d words:" % i, end='')

        for ind in order_centroids[i, :6]:  # replace 6 with n words per cluster
            print(' %s' % terms[ind], end=',')
        print()  # add whitespace
        print()  # add whitespace
