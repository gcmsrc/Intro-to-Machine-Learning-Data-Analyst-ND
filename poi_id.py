#!/usr/bin/python

import sys
import pickle

from tester import dump_classifier_and_data

sys.path.append("tools/")
from feature_format import featureFormat, targetFeatureSplit
import dict_parser



### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
				 'to_messages',
				 'expenses',
				 'from_poi_to_this_person',
				 'shared_with_poi_ratio',
				 'shared_receipt_with_poi',
				 'other',
				 'to_poi_ratio',
				 'bonus',
				 'total_stock_value',
				 'restricted_stock',
				 'salary',
				 'sqrt_wealth',
				 'total_payments',
				 'exercised_stock_options',
				 'sqrt_exercised_stock_options']

### Load the dictionary containing the dataset
with open("data/final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
outliers = ['TOTAL',
            'SHAPIRO RICHARD S',
            'KAMINSKI WINCENTY J',
            'KEAN STEVEN J',
            'LOCKHART EUGENE E',
            'THE TRAVEL AGENCY IN THE PARK']
### Task 3: Create new feature(s)
# Define adder dictionary for adding new variables
adder_dictionary = {'ratio' :
							{'exercised_ratio':
										['exercised_stock_options',
										'total_stock_value'],
							'from_poi_ratio' :
										['from_poi_to_this_person',
										'to_messages'],
                           	'to_poi_ratio' :
                           				['from_this_person_to_poi',
                           				'from_messages'],
                           	'shared_with_poi_ratio' :
                           				['shared_receipt_with_poi',
                           				'to_messages']},
                    'additive' :
                    		{'wealth' :
                    					['salary',
                    					'total_payments',
                    					'bonus',
                    					'total_stock_value',
                    					'expenses',
                    					'other',
                    					'restricted_stock']}}

### Store to my_dataset for easy export below.
my_dataset = dict_parser.parse(data_dict,
							   outliers,
							   adder_dictionary,
							   log_sqrt=['wealth',
							   'exercised_stock_options'])

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### See notebook Classification Full in ../notebooks


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
clf = Pipeline(steps=[('minmaxscaler',
					   MinMaxScaler(copy=True,
					   				feature_range=(0, 1))),
					  ('logisticregression',
					  LogisticRegression(C=10000,
									     class_weight='balanced',
									     dual=False,
									     fit_intercept=True,
									     intercept_scaling=1,
									     max_iter=100,
									     multi_class='ovr',
									     n_jobs=1,
									     penalty='l2',
									     random_state=42,
									     solver='liblinear',
									     tol=0.0001,
									     verbose=0,
									     warm_start=False))])

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### See notebook Classification Full in ../notebooks

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)