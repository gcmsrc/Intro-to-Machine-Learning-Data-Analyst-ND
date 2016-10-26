""" 
    
    The variable_adder Python module adds variable to an Enron
    pandas dataframe as per the instruction provided into an 
    adder dictionary. An adder dictionary is formatted as follows:

    adder_dictionary = {'ratio' :
            {'bonus_ratio': ['bonus', 'salary'],
            'expenses_ratio' : ['expenses', 'salary'],
            'payments_ratio' : ['total_payments', 'salary'],
            'from_poi_ratio' : ['from_poi_to_this_person',
                                'to_messages'],
            'to_poi_ratio' : ['from_this_person_to_poi',
                              'from_messages'],
            'shared_with_poi_ratio' : ['shared_receipt_with_poi',
                                                  'to_messages']},
            'additive' : {'wealth' : ['salary',
                                      'bonus',
                                      'total_stock_value']}}

    The adder dictionary above adds a total of 7 variables to a
    dataframe. Variables are of two types.

    Ratio variables are simply the ration between a numerator
    and denominator. For example, the new variable bonus_ratio
    will be the ratio of bonus vs salary.

    Additive variables are the sum of two or more variables.
    The wealth variable will be the sum of salary, bonus and
    total_stock_value.

"""

import pandas as pd
import numpy as np

##################################
# ~~~~~~~~~~~~~~ ADD RATIO ~~~~~~~

def add_ratio(dataframe, new_var, numerator, denominator):
    
    """
        
        This function add a ratio variable to a dataframe.

        Args:
            - dataframe: a pandas dataframe
            - new_var: a string, i.e. the name of the new
                       variable
            - numberator: a string, i.e. the name of the variable
                          which is the numerator of the ratio
                          variable
            - denominator: a string, i.e. the name of the variable
                           which is the denominator of the ratio
                           variable

        Returns:
            - dataframe: a pandas dataframe after having added the
                         ratio variable
    
    """
    
    dataframe[new_var] = dataframe[numerator] / dataframe[denominator]
    
    return dataframe

##################################
# ~~~~~~~~~~~~~~ ADD ADDITIVE ~~~~

def add_additive(dataframe, new_var, variables, replace_nan = True):
    
    """
    
        This function add an additive variable to a dataframe.

        Args:
            - dataframe: a pandas dataframe
            - new_var: a string, i.e. the name of the new
                       variable
            - variables: a list of variables to be added
            - replace_nan: a boolean. If true, NaN values are
                           replaced by 0 (so that the sum is not
                            NaN by default)

        Returns:
            - dataframe: a pandas dataframe after having added the
                         additive variable
    
    """
    
    # Initialise dataset[new_var] as a series of all zero values
    dataframe[new_var] = 0
    
    if replace_nan:
        
        for variable in variables:
            
            dataframe[new_var] = dataframe[new_var] \
                                 + dataframe[variable].fillna(0)
        
    else:
        
        for variable in variables:
            
            dataframe[new_var] = dataframe[new_var] \
                                 + dataframe[variable]
            
    return dataframe


##################################
# ~~~~~~~~~~~~~~ LOG_SQRT_ADDER
def log_sqrt_adder(dataframe, variable):

    """

        This function add the log10 tranformed and sqrt-transformed
        of a given variables to a specific dataframe.

        Args:
            - dataframe: a pandas dataframe
            - variable: a string, i.e. the name of the variable for
                        which log10 and sqrt transformations will
                        be added

        Returns:
            - dataframe: a pandas dataframe after having added the
                         log10 and sqrt variables

    """

    dataframe['log_' + variable] = np.log10(dataframe[variable])
    dataframe['sqrt_' + variable] = np.sqrt(dataframe[variable])

    return dataframe



##################################
# ~~~~~~~~~~~~~~ ADD ALL ~~~~~~~~~
def add_all(dataframe, adder_dictionary, log_sqrt):
    
    """
        This function returns a pandas dataframe after having
        added new features as specified in the adder dictionary.

        Args:
            - dataframe: a pandas dataframe
            - adder_dictionary: a Python dictionary specifying
                                instructions on variables to add
            - log_sqrt: a Python of variables for which log10 and
                        sqrt transformations are added to the
                        dataframe

        Returns:
            - dataframe: a pandas dataframe with the added variables
    
    """
    
    for adder, instruction in adder_dictionary.items():
        
        if adder == 'ratio':
            
            for key, variables in instruction.items():
                
                dataframe = add_ratio(dataframe,
                                      key,
                                      variables[0],
                                      variables[1])
                
        if adder == 'additive':
            
            for key, variables in instruction.items():
                
                dataframe = add_additive(dataframe,
                                         key,
                                         variables)
                
                    
    if log_sqrt:

        for variable in log_sqrt:

            dataframe = log_sqrt_adder(dataframe,
                                       variable)

    # Replace -inf values with zero
    dataframe = dataframe.replace(-np.inf, 0)

    return dataframe
