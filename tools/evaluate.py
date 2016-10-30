"""

	This module contains scripts for evaluating an algorithm, i.e.
	calculating metrics over 1,000 stratified shuffled samples.

"""

import sys
import numpy as np
import pandas as pd
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, \
                            recall_score, f1_score, roc_auc_score

#####################################
### EVALUATE SINGLE CLASSIFIER ###
# This functione evaluate a classifier using scores on label
# 1, i.e. metrics are calculated on the POI 1 values only.

def eval_clf(clf_best,
			 features,
			 labels):
    
    """
    
        This function evaluates a classification algorithm over
        a series of stratified samples.
        
        Args:
            - clf_best: a classifier
            - features, labels: array of features,
            					labels extracted from the dataset.
                                
        Returns:
            - list of scores
        
    """
    
    # Define cv object
    cv = StratifiedShuffleSplit(labels, 1000, random_state=42)
    
    accuracy = []
    precision = []
    recall = []
    f1 = []
    
    for train_idx, test_idx in cv:
        
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
            
        
        ### fit the classifier using training set,
        #and test on test set
        clf_best.fit(features_train, labels_train)
        predictions = clf_best.predict(features_test)
        probabs = clf_best.predict_proba(features_test)
        
        accuracy.append(accuracy_score(labels_test, predictions))
        precision.append(precision_score(labels_test, predictions))
        recall.append(recall_score(labels_test, predictions))
        f1.append(f1_score(labels_test, predictions))
        
    accuracy = np.array(accuracy).mean()
    precision = np.array(precision).mean()
    recall = np.array(recall).mean()
    f1 = np.array(f1).mean()
    
    return [accuracy, precision, recall, f1]

#####################################
### EVALUATE SINGLE CLASSIFIER ###
# This functione evaluate a classifier using global scores
# as in tester.py


def eval_clf_tester(clf_best,
					features,
					labels):
    
    """
    
        This function evaluates a classification algorithm over
        a series of stratified samples.
        
        Args:
            - clf_best: a classifier
            - features, labels: array of features,
            					labels extracted from the dataset.
                                
        Returns:
            - list of scores, where metrics are calculated as in
              tester.py
        
    """

    # Define cv object
    cv = StratifiedShuffleSplit(labels, 1000, random_state=42)

    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        
        # fit the classifier using training set,
        # and test on test set
        clf_best.fit(features_train, labels_train)
        predictions = clf_best.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            
    
    total_predictions = true_negatives \
    					+ false_negatives \
    					+ false_positives \
    					+ true_positives

    accuracy = 1.0*(true_positives + true_negatives)\
    		   /total_predictions

    precision = 1.0*true_positives\
    			/(true_positives+false_positives)

    recall = 1.0*true_positives\
    		 /(true_positives+false_negatives)

    f1 = 2.0 * true_positives\
    	 /(2*true_positives + false_positives+false_negatives)

    return [accuracy, precision, recall, f1]


#####################################
### EVALUATE LIST OF CLASSIFIERS ###

def evaluate_clf_list(clfs_list,
                      features,
                      labels,
                      tester):
    
    """
    
        This function performs a cross-validated evalution of a
        list of classifiers.
        
        Args:
            - clf_list: a list of classifiers to evaluate
            - features, labels: array of features, labels extracted
            					from the dataset.
            - tester: a boolean. If true, the evaluation applied
            		  is the one provided in the course using
            		  tester.py
                                
        Returns:
            - clfs_names: list with names of classifiers
            - clfs_score: list with scores after evaluation
    
    """
    
    clfs_names = []
    clfs_scores = []
    
    for clf in clfs_list:
        
        # Extract name of classifier
        clfs_names.append(clf['name'])
        
        sys.stdout.write('%s in progress' % clf['name'])
        
        if tester:
            
            clf_score = eval_clf_tester(clf['best_clf'],
            							features,
            							labels)
        else:

            clf_score = eval_clf(clf['best_clf'],
                                     features,
                                     labels)
        
        sys.stdout.write('...........Completed!\n') 
        
        clfs_scores.append(clf_score)
        
    
    return clfs_names, clfs_scores


#####################################
### RANK CLASSIFIERS ###
# This function provides a ranking of the classifiers,
# after they have been evaluated.

def clf_ranking(clfs_list,
                features,
                labels,
                by = 'f1',
                tester = False):
    
    """
    
        This function returns a ranked pandas dataframe of
        classifiers and their performance metrics.
        
        Args:
            - clf_list: a list of classifiers to evaluate
            - features, labels: array of features, labels extracted
            					from the dataset.
            - by: a string, specifying the metric used for ranking.
    			  f1 is the default.
            - tester: a boolean. If true, the evaluation applied
            		  is the one provided in the course using
            		  tester.py
    
    """
    
    # Evaluate a list of classifiers
    names, scores = evaluate_clf_list(clfs_list,
                                      features = features,
                                      labels = labels,
                                      tester = tester)
    
    # Initialise the ranking dataframe
    ranking = pd.DataFrame(names, columns =  ['name'])
    
    # Add values
    columns = ['accuracy', 'precision', 'recall','f1']
    
    for i in range(0, len(columns)):
        
        ranking[columns[i]] = [x[i] for x in scores]
        
    # Return ranked dataframe
    return ranking.sort_values(by = by, ascending = False)
