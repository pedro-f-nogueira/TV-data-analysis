# coding=utf-8

"""
.. module:: social_network.py
    :synopsis: This module used a Random Forest to determine several categories using program descriptions
               It uses data that was categorized by humans in order to train the algorythm. This data is composed
               by a category of a given program and its clean description.
               It may take a few hours to run.
.. moduleauthor:: Pedro Nogueira <pedro.fig.nogueira@gmail.com>
"""

import os
import sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)
del path

import utilities.utilities as util
import networkx as nx
from pylab import show

try:
    import pygraphviz
    from networkx.drawing.nx_agraph import graphviz_layout
except ImportError:
    try:
        import pydotplus
        from networkx.drawing.nx_pydot import graphviz_layout
    except ImportError:
        raise ImportError("This example needs Graphviz and either "
                          "PyGraphviz or PyDotPlus")


def most_important(graph):
    """
    returns a copy of G with
    the most important nodes
    according to the pagerank
    """
    ranking = nx.betweenness_centrality(graph).items()
    r = [x[1] for x in ranking]
    m = sum(r)/len(r) # mean centrality
    t = m*3 # threshold, we keep only the nodes with 3 times the mean
    Gt = graph.copy()
    for k, v in ranking:
        if v < t:
            Gt.remove_node(k)

    return Gt

print 'Fetching nodes and edges from the database...'

nodes = util.get_df_from_conn('SELECT CHANNEL_N, CHANNEL_NAME, SUM_DURATION_DAYS FROM NODES_CHANNELS_TEST ', 'local')

edges = util.get_df_from_conn('SELECT SOURCES_N, DESTINATION_N, STRENTH, (STRENTH/90) NORM_STRENTH FROM EDGES_CHANNELS_1'
                              ' WHERE STRENTH > 40', 'local')

# We are using a DirectedGraph here since the social network has one-way relationships
# graph = nx.path_graph(4)
graph = nx.DiGraph()

# First thing to do is to add the nodes to the graph
labels = {}
a = 0
for index, row in nodes.iterrows():
    b = row[0]
    graph.add_node(row[0], label=row[1])
    graph.node[b]['state'] = row[1]
    labels[a] = row[1].encode('utf-8')
    a = a + 1

# Then we determine the edges on the nodes
for index, row in edges.iterrows():
    graph.add_edge(row[0], row[1], weight=row[3])

grapht = most_important(graph) # trimming

pos = nx.spring_layout(graph, scale=50)

d = nx.degree(graph)

print 'Printing the social network...'

# draw the nodes and the edges (all)
nx.draw_networkx_nodes(graph, pos, vnode_color='b', alpha=0.2, node_size=[v * 100 for v in d.values()])
nx.draw_networkx_edges(graph, pos, valpha=0.1, width=0.1)

# draw the most important nodes with a different style
nx.draw_networkx_nodes(grapht, pos, node_color='r', alpha=0.4, node_size=[v * 100 for v in d.values()])

# get the labels
node_labels = nx.get_node_attributes(graph, 'state')
nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=10)

show()
