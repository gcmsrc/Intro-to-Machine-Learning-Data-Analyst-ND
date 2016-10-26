""" 
    
    The dict_parser Python module is a module that allows to
    parse the Enron data dictionary by:

        - removing outliers;
        - adding new features (via the variable_adder module).

    The main function is parse_dictionary, which ultimately returns
    a parsed Python data dictionary, after having performed a
    series of operations on a pandas datafarme.

    The module also imports pandas, numpy and math.

"""

import pandas as pd
import numpy as np
import math
import variable_adder

###############################################################

### CONVERSION TO PANDAS DATAFRAME ###

# The two functions below allows to convert a Enron data
# dictionary into a pandas dataframe

def extract_fields_from_dict(data_dict):

    """

        This function takes a Python dictionary and extract
        the fields that will be the variable names of the
        pandas dataframe created in this module.

        Args:
            - data_dict: a Python dictionary

        Returns:
            - fields: a list of variable names

    """

    for _, sub_dict in data_dict.items():

        fields = ['name']

        for key, value in sub_dict.items():
            fields.append(key)

        break

    return fields

def convert_into_df(data_dict):

    """

        This function takes a Python dictionary and converts it
        into a pandas dataframe.

        Args:
            - data_dict: a Python dictionary

        Returns:
            - dataframe: a pandas dataframe

        This function takes a data dictionary formatted as the
        Enron one and returns a pandas dataframe.

    """

    # Extract fields to be used as columns in the dataframe
    fields = extract_fields_from_dict(data_dict)


    # Initialise the values list, i.e. a list of lists
    # for all observations
    values = []

    for name, dictionary in data_dict.items():

        # Initialise the observation list and add name
        # values immediately
        obs_list = [name]

        for key, value in dictionary.items():

            if key == 'email_address':
                obs_list.append(value)
            else:
                obs_list.append(float(value))

        values.append(obs_list)

    # Initialise pandas dataframe
    dataframe = pd.DataFrame(values, columns = fields)

    return dataframe

###############################################################

### REMOVE OUTLIER ###

# The funtion remove_outliers is used to return a dataframe
# where particular names have been removed.

def remove_outliers(dataframe, outliers):
    
    """
        
        This function remove outlier values from a pandas
        dataframe.

        Args:
            - dataframe: a pandas DataFrame
            - outliers: a list, i.e. the names of the values
                        to be removed from the DataFrame

        Returns:
            - dataframe: a pandas dataframe after having removed
                         the outliers
    """
    
    # Define slicing array
    slicer = np.invert(dataframe['name'].isin(outliers))
    
    dataframe = dataframe[slicer]
    # return sclider outliers
    return dataframe


###############################################################

### WRITE DICTIONARY ###

# The funtion write_dictionary converts a pandas dataframe
# into a Python dictionary formatted as the Enron dataset.

def write_dictionary(dataframe):
    
    """

        This funcion creates a Python dictionary formatted
        as the Enron dataset.

        Args:
            - dataframe: a pandas DataFrame

        Returns:
            - data_dict: a Python dictionary

    """
    
    # Initialise the dictionary
    data_dict = {}
    
    # Extract variable names
    variables = dataframe.columns.values.tolist()
    
    # For loop to add items
    for observation in dataframe.values.tolist():
        
        # Initialise the single observation dictionary
        obs_dict = {}
        
        for i in range(1, len(variables)):
            
            value = observation[i]
            
            if isinstance(value, float):
                
                if math.isnan(value):
                    
                    obs_dict[variables[i]] = 'NaN'
                    
                else:
                    
                    obs_dict[variables[i]] = observation[i]
            
            else:
                
                obs_dict[variables[i]] = observation[i]
            
            
        # Add item to data_dict
        data_dict[observation[0]] = obs_dict
    
    return data_dict
    
###############################################################

### PARSE DICTIONARY ###

# The parse_dictionary function includes all the functions above.
# It takes a data dictionary formatted as tge Enron one, and
# transform it into a new dictionary without outliers and 
# with added features.

def parse_dictionary(data_dict,
                     outliers,
                     adder_dictionary, 
                     log_sqrt = None):
    
    """
        
        This is a non-plus-ultra function, i.e. it does it all.
        It takes the Enron dataset, formatted as a Python
        dictionary and it:
            - removes the outliers
            - add new features
            - return a dictionary again.

        Args:
            - data_dict: a Python dictionary
            - outliers: a Python list
            - adder_dictionary: a Python dictionary
            - log_sqrt: a Python list. By default it is set to None

        Returns:
            - data_dict: a Python dictionary
        
    
    """
    
    # Convert data_dict into dataframe
    dataframe = convert_into_df(data_dict)
    
    # Remove outliers
    dataframe = remove_outliers(dataframe, outliers)
    
    # Add variables
    dataframe = variable_adder.add_all(dataframe,
                                       adder_dictionary,
                                       log_sqrt)
    
    # Returnd dictionary
    data_dict = write_dictionary(dataframe)
    
    return data_dict

###############################################################

### ALGORITHM FIELDS ###

def extract_fields_for_ml(data_dict,
                          threshold = 0.5,
                          exclude = ['name',
                                     'email_address',
                                     'poi']):

    """

        This function extracts a list of features in the
        Enron dataset. The features are those for which the
        proportion of missing values is below a certain
        threshold.

        Args:
            - data_dict: a Python dictionary
            - threshold: a float, by default set to 0.5
            - exclude: a Python list, i.e. variables
                       which are not included in the analysis.
                       By default, they are 'name', 'email_address',
                       and 'poi'.

        Returns:
            - var_list: a Python list.

    """

    # Extract fields
    fields = extract_fields_from_dict(data_dict)

    # Extract variables to be used in the analysis
    variables = [x for x in fields if x not in exclude]

    # Create dataset
    dataset = convert_into_df(data_dict)

    # Create pandas dataframe
    nan_df = pd.DataFrame(variables, columns=['variable'])

    # Calculate percentage of NaN
    nan_df['nan_perc'] = [(1 - (float(dataset[x].count()) \
        / len(dataset))) for x in nan_df['variable']]

    # List of variables
    var_list = ['poi'] \
        + nan_df[nan_df['nan_perc'] < threshold]['variable'].tolist()

    return var_list


