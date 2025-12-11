
# This python script contains the all the general helper functions used in the notebook.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import fuzzywuzzy to correct the typos in 'Brand', 'fuelType' and 'transmission'
from fuzzywuzzy import fuzz

# Import get_close_matches to identify and group similar words for typo correction in 'model'
from difflib import get_close_matches

# Import cross_validate to evaluate models
from sklearn.model_selection import cross_validate

# Import to measure execution time in model evaluation
import time

# Import to test the log price transformation
from sklearn.compose import TransformedTargetRegressor


###------------------------------------------------------------------------------------------------------------###
###--------------------------- FUNCTIONS USED FOR DATA UNDERSTANDING: -----------------------------------------###
###------------------------------------------------------------------------------------------------------------###

def color_pvalues(val):

    """
    Applies color formatting to p-values for visualization in tables.

    The function returns a string that can be used with pandas 'DataFrame.style' to visually indicate statistical significance:
    - Green background if p-value < 0.05, indicating a significant association.
    - Red background if p-value >= 0.05, indicating no significant association.
    - Black text color and a black border are applied in both cases.

    Parameters
    ----------
    val : float
        The p-value to be evaluated.

    Returns
    -------
    str
        A string that can be used to style a pandas DataFrame cell.
    """

    if val < 0.05:
        return 'background-color: lightgreen; color:black; border: 1px solid black;'
    else:
        return 'background-color: lightcoral; color:black; border: 1px solid black;'

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def color_cramervalues(val):

    """
    Applies color formatting to Cramér's V values for visualization in tables.

    The function returns a string that can be used with pandas 'DataFrame.style' to visually indicate the strength of association between categorical variables:
    - Green background if Cramér's V > 0.6, indicating a strong association.
    - Red background if Cramér's V <= 0.6, indicating a weak or moderate association.
    - Black text color and a black border are applied in both cases.

    Parameters
    ----------
    val : float
        The Cramér's V value to be evaluated.

    Returns
    -------
    str
        A string that can be used to style a pandas DataFrame cell.
    """

    if val > 0.6:
        return 'background-color: lightgreen; color:black; border: 1px solid black;'
    else:
        return 'background-color: lightcoral; color:black; border: 1px solid black;'


###------------------------------------------------------------------------------------------------------------###
###--------------------------- FUNCTIONS TO CORRECT TYPOS IN CATEGORICAL FEATURES: ----------------------------###
###------------------------------------------------------------------------------------------------------------###


def correct_brand_w(df, brand, model):

    '''
    The function will switch the observations 'W' with 'BMW' and 'VW' depending on the correspondence of their
    models in other observations. This function is only applied to one element, one brand and the corresponding model

    Parameters
    -----------
    df : DataFrame
        the DataFrame whose columns are to be fixed
        
    brand : string
        the brand 

    model : string
        the corresponding model

    
    Returns
    -----------
    brand : string
        correct model, which will be 'BMW' or 'VW' if the brand is 'W', and the input brand otherwise
    

    '''

    # If the brand is 'w' and its a string
    if isinstance(brand, str) and brand == 'W':

        # For cicle to go over the brands and corresponding models in the DataFrame
        for brand_in_column, model_in_column in zip(df['Brand'], df['model']):

            # If the same model is found, then return the corresponding brand
            if isinstance(brand_in_column, str) and model_in_column == model and brand_in_column != 'W':
                return brand_in_column
            
    # If the brand is not 'W', it remains the same           
    return brand



#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def create_clusters(df, unique_words, column, threshold=85): 
      
    """
    Groups similar strings into clusters and creates a mapping to the most frequent name in each cluster.

    The function checks each element in `unique_words` and groups together all elements whose similarity (calculated using fuzz.WRatio) is greater than or equal to `threshold`. 
    Then, for each cluster, it selects the most common name within that cluster as the representative name, creating a mapping for standardizing the names.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the original data.

    unique_words : list
        List of strings (e.g., brand names) to be clustered.
    
    column : str
        Name of the column in the DataFrame `X_train` where the original elements are located.
    
    threshold : int, default=85
        Minimum similarity percentage (0-100) to consider two strings as belonging to the same cluster. The higher the value, the stricter the grouping.

    Returns
    -------
    clusters : list of lists
        List of clusters, where each cluster is a list of similar strings.
    
    mapping : dict
        Dictionary mapping each original string to the representative name of its cluster (the most frequent name in the cluster).
    """
    
    clusters = []
    for word in unique_words:
        found = False
        for cluster in clusters:
            # evaluates if brand is similar to any cluster
            if any(fuzz.WRatio(str(word), str(b)) >= threshold for b in cluster):
                cluster.append(word)
                found = True
                break
        #if it doesn't find a match --> new cluster
        if not found:
            clusters.append([word])

    # Gives the clusters names- chooses the most freq name
    mapping = {}
    counts = df[column].value_counts()
    for cluster in clusters:
        mode = max(cluster, key=lambda x: counts.get(x,0))  #finds the "max" in the cluster according to the key --> mode
        for word in cluster:
            mapping[word] = str(mode)

    return clusters, mapping




#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------





def correct_categorical (mapping, value, threshold=85):
     
    """
    Corrects a categorical value based on a given mapping or fuzzy string matching.

    The function attempts to standardize a categorical `value`:
    1. If the value exists as a key in `mapping`, it returns the corresponding mapped value.
    2. Otherwise, it compares the value with all unique mapped values using fuzzy string matching (WRatio).
    If a match is found with a similarity greater than or equal to `threshold`, it returns the best match.
    3. If no match is found, it returns the original value.

    Parameters
    ----------
    mapping : dict
        Dictionary mapping known categorical values to their standardized form.
    
    value : str
        The categorical value to be corrected.
    
    threshold : int, default=85
        Minimum similarity percentage (0-100) required to consider a fuzzy match valid.

    Returns
    -------
    str
        The corrected categorical value, either from the mapping, from the best fuzzy match, 
        or the original value if no match is found.
    """

    unique_values = list(mapping.keys())
    if value in unique_values:
        return mapping [value] 
    else: 
        correct_values = set(mapping.values())
        value_match = ''
        max_ratio = threshold
        for element in correct_values:
            if (fuzz.WRatio(str(element), str(value)) >= max_ratio):
                max_ratio= fuzz.WRatio(str(element), str(value))
                value_match = element
        if value_match != '': 
            return value_match
        else:
            return value
            


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def similar_models(df, models, threshold=0.85):

    '''
    The function will group similar models together based on a similarity threshold, with the goal to identify potential typos.

    Parameters
    -----------
    df : pandas.DataFrame
        DataFrame containing the original data.

    models : List[str]
        List with all the unique elements in the 'model' column

    threshold : float , default=0.85
        The similarity threshold for grouping models. 

    Returns
    -----------
    similar_groups : List[List[str]]
        A list of lists, where each inner list contains the models that are similar to each other based on the threshold.

    model_mapping : Dict[str, str]
        A dictionary mapping each model in the train set to its most frequent version.
    '''

    # This list, which starts as an empty list, will store the similar groups of strings
    similar_groups = []
    
    # Start a for loop that will go over all the values in models
    for model in models:

        # Transform de list of lists in a unique list with all the values in the sublists
        similar_groups_flat = [item for sublist in similar_groups if sublist is not None for item in sublist]

        if model in similar_groups_flat:

            # If the model is already in similar_groups_flat, then it already has its similarity group, no need to serach for more
            continue
        else:
             
             # Calculate the similarity between model and the other observations and keep the ones with a similarity higher than 0.85
             close_matches = get_close_matches(model, models, cutoff=threshold)

             model_prefix =  model.split(" ")[0] 
             
             # For the models with more than one word it is necessary to evaluate the prefix in order to separate them well
             if " " in model:

                # Only keep the models with the same model code/ prefix. Different model codes belong to different models
                close_matches = [match for match in close_matches if match.split(" ")[0] == model_prefix]

            # Add the close matches to the list of similar groups
             similar_groups.append(close_matches)

    # Calculate counts once before the function
    model_counts = df['model'].value_counts().to_dict()

    # Dictionary to map each model to its correct version
    model_mapping = {}

    # For loop to go over all the clusters 
    for group in similar_groups:

        # The best model of each cluster will be the one with the highest count in the train set, it will be added to the correct_models list
        best_model = max(group, key=lambda x: model_counts.get(x, 0))

        # Map all models in the group to the best model
        for model in group:
            model_mapping[model] = best_model

    return similar_groups, model_mapping




#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def correct_column_model(model, mapping, threshold = 0.85):


    '''
    The function will correct the typos in the 'model' column by mapping each model to its most frequent version in the train set if it
    appears in the train dataset. If the model is an unseen model, it will find the most similar model based on a similarity threshold
    to the list of correct models.

    Parameters
    -----------

    models : String
        The model name to correct.
    
    mapping : Dict[str, str]
        Dictionary mapping each model in the train set to its most frequent version.

    threshold : float, default = 0.85
        The similarity threshold for grouping models.

    Returns
    -----------
    model : String
        The corrected model name, or the original model name if no correction was made.

    '''

    

    # If the element is NA or an empty string then nothing is changed
    if pd.isna(model) or model == ' ':
        return model
    
    # If the model has only one character then it is not possible to associate it with any model so return NA
    elif len(model) == 1:
        return np.nan

    # Transform de list of lists in a unique list with all the values in the sublists
    models_in_train = list(mapping.keys())

    # If the model is in similar_groups_flat, then it is in the train set and can be mapped to its correct version
    if model in models_in_train:
        return mapping[model]
    
    # If the model has not been seen in the train set, then it will calculate the closest match to the list of correct models
    else:

        correct_models = list(set(mapping.values()))

        # Calculate the similarity between model and the correct models, if the one with highest similarity is above the threshold, then return it
        best_match = get_close_matches(model, correct_models, n=1, cutoff = threshold)

        # If there is a best match above the threshold, return it
        if best_match != []:
            return best_match[0]
        
        # If there wasnt a match above the threshold, return the original model
        else:
            return model





###------------------------------------------------------------------------------------------------------------###
### -------------------------------- FUNCTIONS TO IMPUTE MISSING VALUES ---------------------------------------###
###------------------------------------------------------------------------------------------------------------###



### ------- BRAND -------------------------------------------------------------------------------------------------



def impute_brand(row, mapping, global_mode):

    """
    Imputes missing values in the 'Brand_cleaned' column based on a model-to-brand mapping or a global mode value.

    The function checks a single row of a DataFrame:
    1. If 'Brand_cleaned' is missing (NaN):
       - If 'model_cleaned' exists and is present in 'mapping', it returns the corresponding brand.
       - Otherwise, it returns 'global_mode'.
    2. If 'Brand_cleaned' is not missing, it returns the existing value.

    Parameters
    ----------
    row : pandas.Series
        Represents an individual row in a pandas DataFrame.
    
    mapping : dict
        Dictionary mapping cleaned model names to their corresponding brand.
    
    global_mode : str
        The global most frequent brand value, used as an alternative for missing data.

    Returns
    -------
    str
        The imputed brand value for the row.
    """

    if pd.isna(row['Brand_cleaned']):
        if pd.notna(row['model_cleaned']) and row['model_cleaned'] in mapping:
            return mapping[row['model_cleaned']]
        else:
            return global_mode
    return row['Brand_cleaned']



### ------- MODEL -------------------------------------------------------------------------------------------------


def build_model_mappings(df):

    """
    Builds multiple mappings for imputing car models based on available specifications.

    This function creates several mappings of varying detail, which can be used to impute missing values in the 'model_cleaned' column based on combinations of other features.
    Each mapping uses the model's mode for the given feature combination.

    The mappings are built in decreasing order of detail:
    - mapping_5: Uses Brand, Fuel Type, Engine Size, and Transmission.
    - mapping_4: Uses Brand, Fuel Type, and Engine Size.
    - mapping_3: Uses Brand and Fuel Type.
    - mapping_2: Uses Brand and Transmission.
    - mapping_1: Uses Brand and Engine Size.
    - mapping_0: Uses Brand only.

    Parameters
    ----------
    df : pandas.DataFrame
        containing the columns 'Brand_cleaned', 'model_cleaned', 'fuelType_cleaned','engineSize', and 'transmission_cleaned'.

    Returns
    -------
    mapping_5, mapping_4, mapping_3, mapping_2, mapping_1, mapping_0 : dict
        Six dictionaries mapping combinations of features to the most frequent model name.
    """

    mapping_5 = (
        df.dropna(subset=['Brand_cleaned', 'model_cleaned', 'fuelType_cleaned', 'engineSize', 'transmission_cleaned'])
          .groupby(['Brand_cleaned', 'fuelType_cleaned', 'engineSize', 'transmission_cleaned'])['model_cleaned']
          .agg(lambda x: x.mode().iloc[0]).to_dict()
    )
    mapping_4 = (
        df.dropna(subset=['Brand_cleaned', 'model_cleaned', 'fuelType_cleaned', 'engineSize'])
          .groupby(['Brand_cleaned', 'fuelType_cleaned', 'engineSize'])['model_cleaned']
          .agg(lambda x: x.mode().iloc[0]).to_dict()
    )
    mapping_3 = (
        df.dropna(subset=['Brand_cleaned', 'model_cleaned', 'fuelType_cleaned'])
          .groupby(['Brand_cleaned', 'fuelType_cleaned'])['model_cleaned']
          .agg(lambda x: x.mode().iloc[0]).to_dict()
    )
    mapping_2 = (
        df.dropna(subset=['Brand_cleaned', 'model_cleaned', 'transmission_cleaned'])
          .groupby(['Brand_cleaned', 'transmission_cleaned'])['model_cleaned']
          .agg(lambda x: x.mode().iloc[0]).to_dict()
    )
    mapping_1 = (
        df.dropna(subset=['Brand_cleaned', 'model_cleaned', 'engineSize'])
          .groupby(['Brand_cleaned', 'engineSize'])['model_cleaned']
          .agg(lambda x: x.mode().iloc[0]).to_dict()
    )

    mapping_0 = (
        df.dropna(subset=['Brand_cleaned', 'model_cleaned'])
          .groupby(['Brand_cleaned'])['model_cleaned']
          .agg(lambda x: x.mode().iloc[0]).to_dict()
    )

    return mapping_5, mapping_4, mapping_3, mapping_2, mapping_1, mapping_0



def impute_model_flexible(row, maps):

    """
    Imputes missing 'model_cleaned' values using hierarchical mappings of varying detail.

    The function attempts to fill missing model values in a flexible way:
    1. It first tries the most specific mapping (mapping_5) that uses Brand, Fuel Type, Engine Size, and Transmission.
    2. If no match is found, it proceeds to less detailed mappings in the following order:
       mapping_4, mapping_3, mapping_2, mapping_1, mapping_0 (Brand only).
    3. If no mapping contains a match, the original model value is returned.

    Parameters
    ----------
    row : pandas.Series
        Represents an individual row in a pandas DataFrame.
    
    maps : tuple
        A tuple containing six dictionaries (mapping_5, mapping_4, mapping_3, mapping_2, mapping_1, mapping_0).

    Returns
    -------
    str
        The imputed model name if a match is found in any mapping, or the original 'model_cleaned' value if no match is found.
    """
    
    mapping_5, mapping_4, mapping_3, mapping_2, mapping_1, mapping_0 = maps

    # Try the most specific match first
    key5 = (row['Brand_cleaned'], row['fuelType_cleaned'], row['engineSize'], row['transmission_cleaned'])
    key4 = (row['Brand_cleaned'], row['fuelType_cleaned'], row['engineSize'])
    key3 = (row['Brand_cleaned'], row['fuelType_cleaned'])
    key2 = (row['Brand_cleaned'], row['transmission_cleaned'])
    key1 = (row['Brand_cleaned'], row['engineSize'])
    key0 = (row['Brand_cleaned'],)

    if pd.isna(row['model_cleaned']):
        if key5 in mapping_5: return mapping_5[key5]
        if key4 in mapping_4: return mapping_4[key4]
        if key3 in mapping_3: return mapping_3[key3]
        if key2 in mapping_2: return mapping_2[key2]
        if key1 in mapping_1: return mapping_1[key1]
        if key0 in mapping_0: return mapping_0[key0]
    return row['model_cleaned']




### ------- YEAR -------------------------------------------------------------------------------------------------

def build_year_mappings(df):

    """
    Builds hierarchical mappings to impute missing 'year' values based on mileage, tax and mpg.

    The function creates three mappings of decreasing importance for imputing missing year values:
    1. mapping_3: Uses mileage binned to the median year.
    2. mapping_2: Uses only 'tax' to map to the median year.
    3. mapping_1: Uses only 'mpg' to map to the median year.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame used as the basis for constructing the maps.

    Returns
    -------
    mapping_3, mapping_2, mapping_1 : dict
        Three dictionaries mapping feature combinations to the median year value.

    """


    mapping_3 = (
    df.dropna(subset=['year', 'mileage_bin'])
      .groupby('mileage_bin')['year']
      .median()
      .to_dict()
    )
    
    mapping_2 = (
        df.dropna(subset=['year', 'tax'])
        .groupby('tax')['year']
        .median()
        .to_dict()
    )

    mapping_1 = (
        df.dropna(subset=['year', 'mpg'])
        .groupby('mpg')['year']
        .median()
        .to_dict()
    )
    return mapping_3, mapping_2, mapping_1



def impute_year(row, maps, global_median):

    """
    Imputes missing 'year' values based on related features or a global median.

    The function checks a single row and attempts to fill missing 'year' values in the following order:
    1. If 'mileage_bin' exists and is in 'map_mileage', returns the mapped year.
    2. Else if 'tax' exists and is in 'map_tax', returns the mapped year.
    3. Else if 'mpg' exists and is in 'map_mpg', returns the mapped year.
    4. If none of the above conditions are met, returns 'global_median' as a fallback.
    5. If 'year' is not missing, it returns the original value.

    Parameters
    ----------
    row : pandas.Series
        Represents an individual row in a pandas DataFrame.

    maps : tuple
        A tuple containing three dictionaries (map_mileage, map_tax, map_mpg).

    global_median : float
        The global median year.

    Returns
    -------
    float
        The imputed year for the row.

    """

    map_mileage, map_tax, map_mpg = maps

    if pd.isna(row['year']):
        if pd.notna(row['mileage_bin']) and row['mileage_bin'] in map_mileage:
            return map_mileage[row['mileage_bin']]
        
        elif pd.notna(row['tax']) and row['tax'] in map_tax:
            return map_tax[row['tax']]
        
        elif pd.notna(row['mpg']) and row['mpg'] in map_mpg:
            return map_mpg[row['mpg']]
        
        return global_median

    return row['year']



### ------- MILEAGE -------------------------------------------------------------------------------------------------


def impute_mileage(row, mapping):

    """
    Imputes missing 'mileage' values based on the car's year using a precomputed mapping.

    The function checks a single row and fills missing mileage as follows:
    1. If 'mileage' is missing (NaN) and the 'year' exists in the 'mapping', it returns the
       corresponding value from the mapping.
    2. If 'mileage' is not missing, it returns the original value.

    Parameters
    ----------
    row : pandas.Series
        Represents an individual row in a pandas Data Frame.

    mapping : dict
        Dictionary mapping each year to a representative mileage (median).

    Returns
    -------
    float
        The imputed mileage for the row, or the original mileage if it is already present.
    """

    if pd.isna(row['mileage']) and row['year'] in mapping:
        return mapping[row['year']]
    return row['mileage']



### ------- TAX -------------------------------------------------------------------------------------------------



def build_tax_mappings(df):

  """
    Builds hierarchical mappings to impute missing 'tax' values based on model and year.

    The function creates three mappings of decreasing detail for imputing missing tax values:
    1. mapping_3: Uses both 'model_cleaned' and 'year' to map to the median tax.
    2. mapping_2: Uses only 'model_cleaned' to map to the median tax.
    3. mapping_1: Uses only 'year' to map to the median tax.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame used as the basis for constructing the maps.

    Returns
    -------
    mapping_3, mapping_2, mapping_1 : dict
        Three dictionaries mapping feature combinations to the median tax value.
    """

  mapping_3 = (
      df.dropna(subset=['model_cleaned','year', 'tax'])
        .groupby(['model_cleaned', 'year'])['tax']
        .median().to_dict()
  )
  mapping_2 = (
      df.dropna(subset=['model_cleaned', 'tax'])
        .groupby(['model_cleaned'])['tax']
        .median().to_dict()
  )
  mapping_1 = (
      df.dropna(subset=['year', 'tax'])
        .groupby(['year'])['tax']
        .median().to_dict()
  )
  return mapping_3, mapping_2, mapping_1




def impute_tax(row, maps):

    """
    Imputes missing 'tax' values using hierarchical mappings of varying detail.

    The function checks a single row and fills missing tax values in the following order:
    1. If the combination of 'model_cleaned' and 'year' exists in mapping_3, it returns the median tax.
    2. Else if 'model_cleaned' exists in mapping_2, it returns the median tax for that model.
    3. Else if 'year' exists in mapping_1, it returns the median tax for that year.
    4. If tax is not missing, the original 'tax' value is returned.

    Parameters
    ----------
    row : pandas.Series
        Represents an individual row in a pandas Data Frame.
    
    maps : tuple
        A tuple containing three dictionaries (mapping_3, mapping_2, mapping_1).

    Returns
    -------
    float
        The imputed tax value for the row, or the original 'tax' value if its already present.
    """
    
    mapping_3, mapping_2, mapping_1 = maps

    key3 = (row['model_cleaned'], row['year'])
    key2 = (row['model_cleaned'])
    key1 = (row['year'])

    if pd.isna(row['tax']):
        if key3 in mapping_3: return mapping_3[key3]
        if key2 in mapping_2: return mapping_2[key2]
        if key1 in mapping_1: return mapping_1[key1]
    return row['tax']



### ------- FUEL TYPE -------------------------------------------------------------------------------------------------


def build_fuel_mappings(df):

    """
    Builds hierarchical mappings to impute missing 'fuelType_cleaned' values based on model or brand.

    The function creates two mappings for imputing missing fuel type values:
    1. mapping_2: Uses 'model_cleaned' to map to the most frequent fuel type for that model.
    2. mapping_1: Uses 'Brand_cleaned' to map to the most frequent fuel type for that brand.

    Parameters
    ----------
    df : pandas.DataFrame
        The Data Frame used as the basis for constructing the maps.

    Returns
    -------
    mapping_2, mapping_1 : dict
        Two dictionaries mapping features to the most frequent fuel type. 
    """

    mapping_2 = (
        df.dropna(subset=['model_cleaned','fuelType_cleaned'])
          .groupby(['model_cleaned'])['fuelType_cleaned']
          .agg(lambda x: x.mode().iloc[0]).to_dict()
    )
    mapping_1 = (
        df.dropna(subset=['Brand_cleaned', 'fuelType_cleaned'])
          .groupby(['Brand_cleaned'])['fuelType_cleaned']
          .agg(lambda x: x.mode().iloc[0]).to_dict()
    )
    return mapping_2, mapping_1




def impute_fueltype(row, maps):

    """
    Imputes missing 'fuelType_cleaned' values using hierarchical mappings of varying detail.

    The function checks a single row and fills missing fuel type values in the following order:
    1. If 'model_cleaned' exists in mapping_2, it returns the most frequent fuel type for that model.
    2. Else if 'Brand_cleaned' exists in mapping_1, it returns the most frequent fuel type for that brand.
    3. If fuelType is not missing, the original 'fuelType_cleaned' value is returned.

    Parameters
    ----------
    row : pandas.Series
        Represents an individual row in a pandas Data Frame.
    
    maps : tuple
        A tuple containing two dictionaries (mapping_2, mapping_1).

    Returns
    -------
    str
        The imputed fuel type for the row, or the original 'fuelType_cleaned' value if its already present.
    """

    mapping_2, mapping_1 = maps

    key2 = (row['model_cleaned'])
    key1 = (row['Brand_cleaned'])

    if pd.isna(row['fuelType_cleaned']):
        if key2 in mapping_2: return mapping_2[key2]
        if key1 in mapping_1: return mapping_1[key1]
    return row['fuelType_cleaned']




### ------- MPG -------------------------------------------------------------------------------------------------



def build_mpg_mappings(df):

    """
    Builds hierarchical mappings to impute missing 'mpg' values based on model or brand.

    The function creates two mappings for imputing missing mpg values:
    1. mapping_2: Uses 'model_cleaned' to map to the median mpg for that model.
    2. mapping_1: Uses 'Brand_cleaned' to map to the median mpg for that brand.

    Parameters
    ----------
    df : pandas.DataFrame
        The Data Frame used as the basis for constructing the maps.

    Returns
    -------
    mapping_2, mapping_1 : dict
        Two dictionaries mapping features to the median mpg.
    """

    mapping_2 = (
        df.dropna(subset=['model_cleaned','mpg'])
          .groupby(['model_cleaned'])['mpg']
          .median().to_dict()
    )
    mapping_1 = (
        df.dropna(subset=['Brand_cleaned', 'mpg'])
          .groupby(['Brand_cleaned'])['mpg']
          .median().to_dict()
    )
    return mapping_2, mapping_1




def impute_mpg(row, maps):

    """
    Imputes missing 'mpg' values using hierarchical mappings of varying detail.

    The function checks a single row and fills missing mpg values in the following order:
    1. If 'model_cleaned' exists in mapping_2, it returns the median mpg for that model.
    2. Else if 'Brand_cleaned' exists in mapping_1, it returns the median mpg for that brand.
    3. If 'mpg' is not missing, the original 'mpg' value is returned.

    Parameters
    ----------
    row : pandas.Series
        Represents an individual row in a pandas Data Frame.
    
    maps : tuple
        A tuple containing two dictionaries (mapping_2, mapping_1).

    Returns
    -------
    float
        The imputed mpg value for the row, or the original 'mpg' value if its already present.
    """

    mapping_2, mapping_1 = maps

    key2 = (row['model_cleaned'])
    key1 = (row['Brand_cleaned'])

    if pd.isna(row['mpg']):
        if key2 in mapping_2: return mapping_2[key2]
        if key1 in mapping_1: return mapping_1[key1]
    return row['mpg']




### ------- ENGINE SIZE -------------------------------------------------------------------------------------------------




def build_engine_mappings(df):

    """
    Builds hierarchical mappings to impute missing 'engineSize' values based on model or brand.

    The function creates two mappings for imputing missing engine size values:
    1. mapping_2: Uses 'model_cleaned' to map to the median engine size for that model.
    2. mapping_1: Uses 'Brand_cleaned' to map to the median engine size for that brand.

    Parameters
    ----------
    df : pandas.DataFrame
        The Data Frame used as basis for constructing the maps.

    Returns
    -------
    mapping_2, mapping_1 : dict
        Two dictionaries mapping features to the median engine size.
    """

    mapping_2 = (
        df.dropna(subset=['model_cleaned','engineSize'])
          .groupby(['model_cleaned'])['engineSize']
          .median().to_dict()
    )
    mapping_1 = (
        df.dropna(subset=['Brand_cleaned', 'engineSize'])
          .groupby(['Brand_cleaned'])['engineSize']
          .median().to_dict()
    )
    return mapping_2, mapping_1




def impute_engine(row, maps):

    """
    Imputes missing 'engineSize' values using hierarchical mappings of varying detail.

    The function checks a single row and fills missing engine size values in the following order:
    1. If 'model_cleaned' exists in mapping_2, it returns the median engine size for that model.
    2. Else if 'Brand_cleaned' exists in mapping_1, it returns the median engine size for that brand.
    3. If engineSize is not missing, the original 'engineSize' value is returned.

    Parameters
    ----------
    row : pandas.Series
        Represents an individual row in a Data Frame.
    
    maps : tuple
        A tuple containing two dictionaries (mapping_2, mapping_1).

    Returns
    -------
    float
        The imputed engine size for the row, or the original 'engineSize' value if the value its already present.
    """

    mapping_2, mapping_1 = maps

    key2 = (row['model_cleaned'])
    key1 = (row['Brand_cleaned'])

    if pd.isna(row['engineSize']):
        if key2 in mapping_2: return mapping_2[key2]
        if key1 in mapping_1: return mapping_1[key1]
    return row['engineSize']



### ------- PAINT QUALITY -------------------------------------------------------------------------------------------------



def build_paint_mappings(df):

    """
    Builds hierarchical mappings to impute missing 'paintQuality%' values based on model and year.

    The function creates two mappings of decreasing detail for imputing missing paint quality values:
    1. mapping_2: Uses 'model_cleaned' and 'year' to map to the median paint quality for that model and year.
    2. mapping_1: Uses 'year' to map to the median paint quality for that year.

    Parameters
    ----------
    df : pandas.DataFrame
        The Data Frame used as basis for constructig the maps.

    Returns
    -------
    mapping_2, mapping_1 : dict
        Two dictionaries mapping feature combinations to the median paint quality percentage. 
    """

    mapping_2 = (
        df.dropna(subset=['model_cleaned','year', 'paintQuality%'])
          .groupby(['model_cleaned', 'year'])['paintQuality%']
          .median().to_dict()
    )
    mapping_1 = (
        df.dropna(subset=['year', 'paintQuality%'])
          .groupby(['year'])['paintQuality%']
          .median().to_dict()
    )
    return mapping_2, mapping_1




def impute_paint(row, maps):

    """
    Imputes missing 'paintQuality%' values using hierarchical mappings of varying detail.

    The function checks a single row and fills missing paint quality values in the following order:
    1. If the combination of 'model_cleaned' and 'year' exists in mapping_2, it returns the median paint quality for that combination.
    2. Else if 'year' exists in mapping_1, it returns the median paint quality for that year.
    3. If paintQuality is not missing, the original 'paintQuality%' value is returned.

    Parameters
    ----------
    row : pandas.Series
        Represents an individual row in a Data Frame.
    
    maps : tuple
        A tuple containing two dictionaries (mapping_2, mapping_1).

    Returns
    -------
    float
        The imputed paint quality percentage for the row, or the original 'paintQuality%' value if the value is already present.
    """

    mapping_2, mapping_1 = maps

    key2 = (row['model_cleaned'], row['year'])
    key1 = (row['year'])

    if pd.isna(row['paintQuality%']):
        if key2 in mapping_2: return mapping_2[key2]
        if key1 in mapping_1: return mapping_1[key1]
    return row['paintQuality%']




### ------- PREVIOUS OWNERS -------------------------------------------------------------------------------------------------


def build_owners_mappings(df):

    """
    Builds hierarchical mappings to impute missing 'previousOwners' values based on model and year.

    The function creates two mappings of decreasing detail for imputing missing previous owners:
    1. mapping_2: Uses 'model_cleaned' and 'year' to map to the median number of previous owners for that combination.
    2. mapping_1: Uses 'year' to map to the median number of previous owners for that year.

    Parameters
    ----------
    df : pandas.DataFrame
        The Data Frame used as basis for constructing the maps.

    Returns
    -------
    mapping_2, mapping_1 : dict
        Two dictionaries mapping feature combinations to the median number of previous owners. 
    """

    mapping_2 = (
        df.dropna(subset=['model_cleaned','year', 'previousOwners'])
          .groupby(['model_cleaned', 'year'])['previousOwners']
          .median().to_dict()
    )
    mapping_1 = (
        df.dropna(subset=['year', 'previousOwners'])
          .groupby(['year'])['previousOwners']
          .median().to_dict()
    )
    return mapping_2, mapping_1




def impute_owners(row, maps):

    """
    Imputes missing 'previousOwners' values using hierarchical mappings of varying detail.

    The function checks a single row and fills missing previous owners values in the following order:
    1. If the combination of 'model_cleaned' and 'year' exists in mapping_2, it returns the median number of previous owners for that combination.
    2. Else if 'year' exists in mapping_1, it returns the median number of previous owners for that year.
    3. If previousOwners is not missing, the original 'previousOwners' value is returned.

    Parameters
    ----------
    row : pandas.Series
        Represents an individual row in a Data Frame.
    
    maps : tuple
        A tuple containing two dictionaries (mapping_2, mapping_1).

    Returns
    -------
    float
        The imputed number of previous owners for the row, or the original 'previousOwners' value if the value is already present.
    """

    mapping_2, mapping_1 = maps

    key2 = (row['model_cleaned'], row['year'])
    key1 = (row['year'])

    if pd.isna(row['previousOwners']):
        if key2 in mapping_2: return mapping_2[key2]
        if key1 in mapping_1: return mapping_1[key1]
    return row['previousOwners']



### ------- TRANSMISSION -------------------------------------------------------------------------------------------------



def build_transmission_mappings(df):

    """
    Builds hierarchical mappings to impute missing 'transmission_cleaned' values based on model or brand.

    The function creates two mappings for imputing missing transmission types:
    1. mapping_2: Uses 'model_cleaned' to map to the most frequent transmission for that model.
    2. mapping_1: Uses 'Brand_cleaned' to map to the most frequent transmission for that brand.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame used as basis for constructing the maps.

    Returns
    -------
    mapping_2, mapping_1 : dict
        Two dictionaries mapping features to the most frequent transmission type. 
    """

    mapping_2 = (
        df.dropna(subset=['model_cleaned','transmission_cleaned'])
          .groupby(['model_cleaned'])['transmission_cleaned']
          .agg(lambda x: x.mode().iloc[0]).to_dict()
    )
    mapping_1 = (
        df.dropna(subset=['Brand_cleaned', 'transmission_cleaned'])
          .groupby(['Brand_cleaned'])['transmission_cleaned']
          .agg(lambda x: x.mode().iloc[0]).to_dict()
    )
    return mapping_2, mapping_1



def impute_transmission(row, maps):

    """
    Imputes missing 'transmission_cleaned' values using hierarchical mappings of varying detail.

    The function checks a single row and fills missing transmission values in the following order:
    1. If 'model_cleaned' exists in mapping_2, it returns the most frequent transmission for that model.
    2. Else if 'Brand_cleaned' exists in mapping_1, it returns the most frequent transmission for that brand.
    3. If transmission is not missing, the original 'transmission_cleaned' value is returned.

    Parameters
    ----------
    row : pandas.Series
        Represents an individual row in a pandas Data Frame.
    
    maps : tuple
        A tuple containing two dictionaries (mapping_2, mapping_1).
    Returns
    -------
    str
        The imputed transmission value for the row, or the original 'transmission_cleaned' value if the value is already present.
    """

    mapping_2, mapping_1 = maps

    key2 = (row['model_cleaned'])
    key1 = (row['Brand_cleaned'])

    if pd.isna(row['transmission_cleaned']):
        if key2 in mapping_2: return mapping_2[key2]
        if key1 in mapping_1: return mapping_1[key1]
    return row['transmission_cleaned']

###------------------------------------------------------------------------------------------------------------###
###--------------------------- FUNCTION FOR ABLATION: -------------------------------------------------------###
###------------------------------------------------------------------------------------------------------------###

def evaluate(model, X, y, cv, scoring):

    """
    Evaluates a regression model using cross-validation and returns
    aggregated performance metrics for both training and validation sets.

    Parameters
    ----------
    model : estimator object
        Our pipeline with the preprocessing steps and the regressor.

    X : pandas.DataFrame
        Feature matrix used for training and validation.

    y : pandas.Series
        Target variable (price).

    cv : cross-validation generator
        Determines the cross-validation splitting strategy.

    scoring : dict
        Dictionary defining the scoring metrics to be evaluated during cross-validation.

    Returns
    -------
    tuple
        A tuple containing:
        - train_r2 : float
            Mean R² score on the training set across all folds.
        - test_r2 : float
            Mean R² score on the validation set across all folds.
        - train_mae : float
            Mean MAE on the training set across all folds.
        - test_mae : float
            Mean MAE on the validation set across all folds.
        - execution_time : float
            Mean time taken to fit the model across all folds.
    """

    results = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=True) # n_jobs=-1 to use all processors
    
    test_r2 = np.mean(results['test_R2'])
    test_mae = -np.mean(results['test_MAE'])  # Negative because MAE is negative in scoring
    train_r2 = np.mean(results['train_R2'])
    train_mae = -np.mean(results['train_MAE'])
    execution_time = np.mean(results['fit_time'])

    return train_r2, test_r2, train_mae, test_mae, execution_time

