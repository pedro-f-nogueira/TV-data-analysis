# coding=utf-8

"""
.. module:: eval_cluster_clients_2.py
    :synopsis: This module evaluates clusters using ANOVA hypothesis tests
               and tukey test. This module is done after variable selection
               through anova tests in eval_cluster_clients.py

.. moduleauthor:: Pedro Nogueira <pedro.fig.nogueira@gmail.com>
"""

import os
import sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)
del path

import utilities.utilities as util
from scipy import stats
from pandas.tools.plotting import parallel_coordinates
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def determine_p_value(i_p_val):
    """
    Returns true if entry is smaller than 0.05
    """
    if i_p_val < 0.05:
        return 'True'
    else:
        return 'False'


def return_anova(i_group_1, i_group_2, i_group_3, i_group_4, i_var_name):
    """
    Compares the distribution of the three inputs by doing ANOVA analysis.
    The p_value < 0.05 means that the distribution is significantly different
    """
    f_val, p_val = stats.f_oneway(i_group_1, i_group_2, i_group_3, i_group_4)

    if determine_p_value(p_val) != 'True':
        return 'P_value for ' + i_var_name + ' is ' + str(p_val) + ' and the hypothesis is ' \
          + determine_p_value(p_val)
    else:
        return 'Hypothesis is valid'


print 'Fetching the dataframe...'

df = util.get_df_from_conn('SELECT CLUSTER, TOTAL_MINUTES, C_GENERALIST_MINUTES,FRIDAY_MINUTES, '
                           'P_CONTESTS_MINUTES, P_COOKING_MINUTES, P_SPORTS_MINUTES, P_DOCUMENTARIES_MINUTES, '
                           'P_MOVIES_MINUTES, P_KIDS_MINUTES, P_INFORMATION_MINUTES, P_MUSIC_MINUTES, '
                           'P_SOAP_OPERAS_MINUTES, P_OTHERS_MINUTES, P_SERIES_MINUTES'
                           ' FROM T_SEGMENT_USERS_DATA_CLUSTERS_2', 'local')

df_A = df[df['CLUSTER'] == 0]
df_B = df[df['CLUSTER'] == 1]
df_C = df[df['CLUSTER'] == 2]
df_D = df[df['CLUSTER'] == 3]

print 'Computing ANOVA statistics...'

print return_anova(df_A['TOTAL_MINUTES'], df_B['TOTAL_MINUTES'], df_C['TOTAL_MINUTES'],
                   df_D['TOTAL_MINUTES'], 'TOTAL_MINUTES')

print return_anova(df_A['C_GENERALIST_MINUTES'], df_B['C_GENERALIST_MINUTES'], df_C['C_GENERALIST_MINUTES'],
                   df_D['C_GENERALIST_MINUTES'], 'C_GENERALIST_MINUTES')

print return_anova(df_A['FRIDAY_MINUTES'], df_B['FRIDAY_MINUTES'], df_C['FRIDAY_MINUTES'],
                   df_D['FRIDAY_MINUTES'], 'FRIDAY_MINUTES')

print return_anova(df_A['P_CONTESTS_MINUTES'], df_B['P_CONTESTS_MINUTES'], df_C['P_CONTESTS_MINUTES'],
                   df_D['P_CONTESTS_MINUTES'], 'P_CONTESTS_MINUTES')
print return_anova(df_A['P_COOKING_MINUTES'], df_B['P_COOKING_MINUTES'], df_C['P_COOKING_MINUTES'],
                   df_D['P_COOKING_MINUTES'], 'P_COOKING_MINUTES')
print return_anova(df_A['P_SPORTS_MINUTES'], df_B['P_SPORTS_MINUTES'], df_C['P_SPORTS_MINUTES'],
                   df_D['P_SPORTS_MINUTES'], 'P_SPORTS_MINUTES')
print return_anova(df_A['P_DOCUMENTARIES_MINUTES'], df_B['P_DOCUMENTARIES_MINUTES'], df_C['P_DOCUMENTARIES_MINUTES'],
                   df_D['P_DOCUMENTARIES_MINUTES'], 'P_DOCUMENTARIES_MINUTES')
print return_anova(df_A['P_MOVIES_MINUTES'], df_B['P_MOVIES_MINUTES'], df_C['P_MOVIES_MINUTES'],
                   df_D['P_MOVIES_MINUTES'], 'P_MOVIES_MINUTES')
print return_anova(df_A['P_KIDS_MINUTES'], df_B['P_KIDS_MINUTES'], df_C['P_KIDS_MINUTES'],
                   df_D['P_KIDS_MINUTES'], 'P_KIDS_MINUTES')
print return_anova(df_A['P_INFORMATION_MINUTES'], df_B['P_INFORMATION_MINUTES'], df_C['P_INFORMATION_MINUTES'],
                   df_D['P_INFORMATION_MINUTES'], 'P_INFORMATION_MINUTES')
print return_anova(df_A['P_MUSIC_MINUTES'], df_B['P_MUSIC_MINUTES'], df_C['P_MUSIC_MINUTES'],
                   df_D['P_MUSIC_MINUTES'], 'P_MUSIC_MINUTES')
print return_anova(df_A['P_SOAP_OPERAS_MINUTES'], df_B['P_SOAP_OPERAS_MINUTES'], df_C['P_SOAP_OPERAS_MINUTES'],
                   df_D['P_SOAP_OPERAS_MINUTES'], 'P_SOAP_OPERAS_MINUTES')
print return_anova(df_A['P_OTHERS_MINUTES'], df_B['P_OTHERS_MINUTES'], df_C['P_OTHERS_MINUTES'],
                   df_D['P_OTHERS_MINUTES'], 'P_OTHERS_MINUTES')
print return_anova(df_A['P_SERIES_MINUTES'], df_B['P_SERIES_MINUTES'], df_C['P_SERIES_MINUTES'],
                   df_D['P_SERIES_MINUTES'], 'P_SERIES_MINUTES')

print 'Computing Tukey Test...'

# Tukey's studentized range test (HSD) is a test specific to the comparison of all pairs of k independent samples.
# Instead we can run t-tests on all pairs, calculate the p-values and apply one of the p-value corrections
# for multiple testing problems.

print pairwise_tukeyhsd(df['P_COOKING_MINUTES'], df['CLUSTER'])

print 'Printing graphics...'

data = df

plt.figure(num=None, figsize=(20, 12), dpi=80)
parallel_coordinates(data, 'CLUSTER')
plt.show()

print 'Success'

