"""

	This module is just a small utility I have used
	to create a csv of the Enron dataset I can easily
	use in R for performing EDA.

	The module extracts the Enron dataset, removes
	outliers and return a csv file.

"""

import pickle
import pandas as pd

import sys
sys.path.append('../tools')
import dict_parser

####### Open dataset #######
with open("../data/final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

####### Remove outliers #######
outliers = ['TOTAL',
 'SHAPIRO RICHARD S',
 'KAMINSKI WINCENTY J',
 'KEAN STEVEN J',
 'LOCKHART EUGENE E',
 'THE TRAVEL AGENCY IN THE PARK']

for outlier in outliers:
    data_dict.pop(outlier, 0)

####### Convert into pandas df #######
df = dict_parser.convert_into_df(data_dict)

####### Return csv #######
df.to_csv('enron.csv')