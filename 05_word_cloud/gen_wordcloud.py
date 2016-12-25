"""
.. module:: model_cluster_topics.py
    :synopsis: This module generates a wordcloud using a given .txt file

.. moduleauthor:: Pedro Nogueira <pedro.fig.nogueira@gmail.com>
"""
import os
import sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)
del path

from os import path
from wordcloud import WordCloud
import matplotlib.pyplot as plt

d = path.dirname(__file__)

# Read the whole text.
text = open(path.join(d, 'OUTROS.txt')).read()

# Generate a word cloud image
wordcloud = WordCloud().generate(text)

plt.imshow(wordcloud)
plt.axis("off")

# take relative word frequencies into account, lower max_font_size
wordcloud = WordCloud(max_font_size=40, relative_scaling=.5).generate(text)
plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.show()