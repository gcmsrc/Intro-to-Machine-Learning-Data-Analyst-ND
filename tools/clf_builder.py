"""

	This module is used to properly format classifiers
	so that they can be used in sklearn modules.

"""

import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.decomposition import PCA


################################################
### CLASSIFIER FORMATTING ###

def name_extractor(clf, compiler = re.compile(r'(\w+)(\(+).*')):

    """

        This function return the name of the classifier as a string.

        Args:
            - clf: a classifier object
            - compliler: a regex compiler

        Returns:
            - clf_name: a string

    """

    clf_name = compiler.match(str(clf)).group(1)

    return clf_name

    

def dict_builder(clf, params):
    
    """
    
        This function returns the starting dict for a classifier.
        The dictionary contains three key/value pairs:
        
            - classifier
            - params, i.e. a dictionary of the parameters used
              in the optimisation
            
        Args:
            - clf: a classifier object
            - params: a dictionary with the parameters of the
            		  classifier
    
    """
    
    clf_dict = {'name' : name_extractor(clf),
                'classifier' : clf,
                'params': params}
    
    return clf_dict


################################################
### PIPE FORMATTING ###

def pipe_param_builder(step_name, parameter_name):
    
    """
    
        This function simply returns the correctly formatted
        name of a parameter to be used in pipeline when
        optimised in GridSearchCV.
    
    """
    
    parameter_name = step_name + '__' + parameter_name
    
    return parameter_name



def pipe_builder(classifier_dict,
                 pca_bool,
                 n_comps = [2, 3, 4, 5, 6, 7, 8, 9, 10]):
    
    """
    
        This function returns pipeline object and
        a dictionary with the parameters to be used for
        optimisation in GridSearchCV.
        
        Args:
            - classifier_dict: a dictionary with a classifier and
            				   its parameters to be optimised
            - pca_bool: a boolean. If true, PCA is added to
            			the pipeline
            - n_comps: a list of possible number of components
            		   for the PCA analysis
                                
        Returns:
            - pipe: a pipeline object
            - pipe_dict: a dictionary with the parameters
            			 for optimisation.
            
    """
    
    # Create scaler object
    scaler = MinMaxScaler()
    
    if pca_bool:
    
        # Initialise pca
        pca = PCA()
        
        # Create pipeline
        pipe = make_pipeline(scaler,
        					 pca,
        					 classifier_dict['classifier'])
    
    
        # Pipe dictionary
        pipe_dict = {pipe_param_builder(pipe.steps[1][0],
                                        'n_components') : n_comps}
    
        # Add parameters of classifier
        for param, values in classifier_dict['params'].items():
            pipe_dict[pipe_param_builder(pipe.steps[2][0],
                                            param)] = values
            
    else:
        
        # If no pca is applied, then the steps in the pipeline
        # are just two
        
        # Create pipeline
        pipe = make_pipeline(scaler, classifier_dict['classifier'])
        
        # Initialise dictionary
        pipe_dict = {}
        
        # Add parameters of classifier
        for param, values in classifier_dict['params'].items():
            pipe_dict[pipe_param_builder(pipe.steps[1][0],
                                            param)] = values
         
    
    return pipe, pipe_dict