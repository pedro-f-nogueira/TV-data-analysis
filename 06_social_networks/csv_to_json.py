# coding=utf-8

"""
.. module:: csv_to_json.py
    :synopsis: This module transforms a csv to json

.. moduleauthor:: Pedro Nogueira <pedro.fig.nogueira@gmail.com>
"""

import csv
import json

csvfile = open('gephi_nodes_week.csv', 'r')
jsonfile = open('gephi_nodes_week.json', 'w')

fieldnames = ("id", "label", "days", "degree", "modularity_class")
reader = csv.DictReader( csvfile, fieldnames)
for row in reader:
    json.dump(row, jsonfile)
    jsonfile.write('\n')

csvfile = open('gephi_edges_week.csv', 'r')
jsonfile = open('gephi_edges_week.json', 'w')

fieldnames = ("source", "target", "weight")
reader = csv.DictReader( csvfile, fieldnames)
for row in reader:
    json.dump(row, jsonfile)
    jsonfile.write('\n')
