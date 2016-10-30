"""
	
	This module contains optimisation functions, i.e. operations
	that return optimised classifiers, given mutliple values
	for the specific parameters. Optimisation is performed with
	GridSearchCV

"""

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
import clf_builder
import sys


#####################################
### SINGLE CLASSIFIER OPTIMISER ###

def optimise_clf(clf_dict,
				 features,
				 labels,
				 scoring,
         pca_bool = False):
    
    """
        
        This function optimises a classifier using GridSearchCV
        PCA.
        
        Args:
            - clf_dict: a dictionary with a classifier and
            		    its parameters to be optimised
            - features, labels: arrays to fit the optimiser on
            - scoring: a string specifying the scoring algorithm
            - pca_bool: a boolean. If True, pca is applied as
               			part of a pipeline
                               
        Returns:
            - best_params: the parameters of the best estimators
    
    """
    
    # Define cv object
    # I am optimising over 10 splits
    cv = StratifiedShuffleSplit(labels, n_iter=10, random_state=42)
    
    if pca_bool:
        
        # Check if classifier has 'max_features' parameter
        if clf_dict['name'] in ['DecisionTreeClassifier',
                                'RandomForestClassifier']:
            
            clf_dict['params'].pop('max_features', None)
        
        # Extract steps and param grid
        pipe, param_grid = clf_builder.pipe_builder(clf_dict,
                                  pca_bool = pca_bool)

        optimiser = GridSearchCV(pipe,
                                 param_grid,
                                 scoring = scoring,
                                 cv = cv)
    
    else:
        
        # Extract steps and param grid
        pipe, param_grid = clf_builder.pipe_builder(clf_dict,
                                  pca_bool = pca_bool)

        optimiser = GridSearchCV(pipe,
                                 param_grid,
                                 scoring = scoring,
                                 cv = cv)

    
    # Fit the optimiser
    optimiser.fit(features, labels)
    
    return optimiser.best_estimator_, optimiser.best_params_


#####################################
### LIST OPTIMISER ###

def optimise_list(clf_list,
                  features,
                  labels,
                  pca_bool = False,
                  scoring = 'f1'):
    
    """
    
        This function optimise a list of classifiers.
        
        Args:
            - clf_list: a list of classifiers
            - features, labels: arrays (by default features and
            					labels extracted)
            - pca: a Boolean. If True, PCA is performed and added
            	   to a pipeline
            - scoring: string, the scoring algorithm for GridSearch.
                       Default is recall.
    
        Returns:
            - best_params: return a list with the best parameters
            			   for the classifiers
        
    """
    new_clf_list = []
    for clf in clf_list:
        
        sys.stdout.write('%s in progress' % clf['name'])
        
        best_clf, best_params = optimise_clf(clf,
                                             features = features,
                                             labels = labels,
                                             pca_bool = pca_bool,
                                             scoring = scoring)
        
        # Modify clf name is pca is applied
        if pca_bool:
            name = clf['name'] + '__pca'
        else:
            name = clf['name']
        
        
        clf_dict = {'name' : name,
                    'classifier' : clf['classifier'],
                    'best_clf' : best_clf,
                    'best_params' : best_params}
        
        new_clf_list.append(clf_dict)
        
        sys.stdout.write('...........Completed!\n')
        
    return new_clf_list
        
        