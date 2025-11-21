
# This python script contains the functions used to preprocess the data inside the K-fold.

# Import general functions
from functions import *

# Import to divide the train set, encode and scale variables
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import TargetEncoder, StandardScaler, OneHotEncoder


def categorical_treatment (X_train, X_val, df_test):

    """ Treatment of categorical variables with typos. """
    
    ######--------------------------------------------------------------------BRAND--------------------------------------------------------------------######

    # Correct the  'W' values in the columns 'Brand' by applying the function correct_brand_w to all elements in the column
    X_train['Brand'] = X_train.apply(lambda row: correct_brand_w(X_train, row['Brand'], row['model']),axis=1)
    X_val['Brand'] = X_val.apply(lambda row: correct_brand_w(X_val, row['Brand'], row['model']), axis = 1)
    df_test['Brand'] = df_test.apply(lambda row: correct_brand_w(df_test, row['Brand'], row['model']), axis = 1)

    ## The ones that were fixed were all 'VW' so we will assume the same for the cases with null values:
    X_train.loc[X_train['Brand'] =='W' ,'Brand'] = 'VW'

    X_val.loc[X_val['Brand'] =='W' ,'Brand'] = 'VW'

    df_test.loc[df_test['Brand'] =='W' ,'Brand'] = 'VW'

    brands = X_train['Brand'].dropna().unique().tolist()

    clusters, mapping = create_clusters(X_train, brands, 'Brand')

    X_train['Brand_cleaned'] = X_train['Brand'].apply(lambda x: correct_categorical(mapping, x))
    X_val['Brand_cleaned'] = X_val['Brand'].apply(lambda x: correct_categorical(mapping, x))
    df_test['Brand_cleaned'] = df_test['Brand'].apply(lambda x: correct_categorical(mapping, x))

    ######-------------------------------------------------------------TRANSMISSION--------------------------------------------------------------------######

    transmission_types = X_train['transmission'].dropna().unique().tolist()

    clusters, mapping = create_clusters(X_train, transmission_types, 'transmission')

    X_train['transmission_cleaned'] = X_train['transmission'].apply(lambda x: correct_categorical(mapping, x))
    X_val['transmission_cleaned'] = X_val['transmission'].apply(lambda x: correct_categorical(mapping, x))
    df_test['transmission_cleaned'] = df_test['transmission'].apply(lambda x: correct_categorical(mapping, x))

    ######---------------------------------------------------------------FUEL TYPES--------------------------------------------------------------------######

    fuel_types=X_train['fuelType'].dropna().unique().tolist()

    clusters, mapping = create_clusters(X_train, fuel_types, 'fuelType')

    X_train['fuelType_cleaned'] = X_train['fuelType'].apply(lambda x: correct_categorical(mapping, x))
    X_val['fuelType_cleaned'] = X_val['fuelType'].apply(lambda x: correct_categorical(mapping, x))
    df_test['fuelType_cleaned'] =  df_test['fuelType'].apply(lambda x: correct_categorical(mapping, x))

    ######-----------------------------------------------------------------MODELS----------------------------------------------------------------------######

    models= X_train['model'].dropna().unique().tolist()

    ## Fuzzywuzzy wasn't able to group the same models in the column 'model', so for this case we will use get_close_matches from the difflib library.
    clusters = similar_models(models)

    # Calculate counts once before the function
    model_counts = X_train['model'].value_counts().to_dict()

    # List to store the most frequent model in each cluster which will be considered the correct one
    correct_models = []

    #   Dictionary to map each model to its correct version
    model_mapping = {}

    # For loop to go over all the clusters 
    for group in clusters:

        # The best model of each cluster will be the one with the highest count in the train set, it will be added to the correct_models list
        best_model = max(group, key=lambda x: model_counts.get(x, 0))
        correct_models.append(best_model)

        # Map all models in the group to the best model
        for model in group:
            model_mapping[model] = best_model


    X_train['model_cleaned'] = X_train['model'].apply(lambda x: correct_column_model(correct_models, model_mapping, x, clusters))
    X_val['model_cleaned'] = X_val['model'].apply(lambda x: correct_column_model(correct_models, model_mapping, x, clusters))
    df_test['model_cleaned'] =  df_test['model'].apply(lambda x: correct_column_model(correct_models, model_mapping, x,clusters))

    return X_train, X_val, df_test

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def numerical_treatment (X_train, X_val, df_test):

    """ Treatment of numerical variables. """

    ######----------------------------------------------------------------OUTLIERS----------------------------------------------------------------------######

    num_feats = ['year', 'mileage', 'tax', 'mpg', 'paintQuality%', 'previousOwners', 'engineSize']

    # - Year: winsorization at 1th percentile:
    lower_cap_year = X_train['year'].quantile(0.01)
    X_train.loc[X_train['year'] < lower_cap_year,'year'] = lower_cap_year
    X_val.loc[X_val['year'] < lower_cap_year,'year'] = lower_cap_year
    df_test.loc[df_test['year'] < lower_cap_year,'year'] = lower_cap_year

    # - Mileage: winsorization at 99th percentile:
    upper_cap_mile = X_train['mileage'].quantile(0.99)
    X_train.loc[X_train['mileage'] > upper_cap_mile,'mileage'] = upper_cap_mile
    X_val.loc[X_val['mileage'] > upper_cap_mile,'mileage'] = upper_cap_mile
    df_test.loc[df_test['mileage'] > upper_cap_mile,'mileage'] = upper_cap_mile

    #- Tax: winsorization at 1th and 99th percentile
    upper_cap_tax = X_train['tax'].quantile(0.99)
    X_train.loc[X_train['tax'] > upper_cap_tax,'tax'] = upper_cap_tax
    X_val.loc[X_val['tax'] > upper_cap_tax,'tax'] = upper_cap_tax
    df_test.loc[df_test['tax'] > upper_cap_tax,'tax'] = upper_cap_tax

    lower_cap_tax = X_train['tax'].quantile(0.01)
    X_train.loc[X_train['tax'] < lower_cap_tax,'tax'] = lower_cap_tax
    X_val.loc[X_val['tax'] < lower_cap_tax,'tax'] = lower_cap_tax
    df_test.loc[df_test['tax'] < lower_cap_tax,'tax'] = lower_cap_tax

    # - mpg: winsorization at 99th percentile
    upper_cap_mpg = X_train['mpg'].quantile(0.99)
    X_train.loc[X_train['mpg'] > upper_cap_mpg,'mpg'] = upper_cap_mpg
    X_val.loc[X_val['mpg'] > upper_cap_mpg,'mpg'] = upper_cap_mpg
    df_test.loc[df_test['mpg'] > upper_cap_mpg,'mpg'] = upper_cap_mpg

    # - engineSize: winsorization at 99th percentile
    upper_cap_engine = X_train['engineSize'].quantile(0.99)
    X_train.loc[X_train['engineSize'] > upper_cap_engine,'engineSize'] = upper_cap_engine
    X_val.loc[X_val['engineSize'] > upper_cap_engine,'engineSize'] = upper_cap_engine
    df_test.loc[df_test['engineSize'] > upper_cap_engine,'engineSize'] = upper_cap_engine
   
    ######------------------------------------------------hasDamage boolean typecasting---------------------------------------------------------------######
    ## Since hasDamage is described as a boolean variable in the metadata, 
    ## we will be assuming that cars with zero values in this variable have no damage and that those with missing values are damaged.
   
    X_train['hasDamage'].isna().replace(True, inplace=True)
    X_val['hasDamage'].isna().replace(True, inplace=True)
    df_test['hasDamage'].isna().replace(True, inplace=True)

    X_train['hasDamage'] = X_train['hasDamage'].astype(bool)
    X_val['hasDamage'] = X_val['hasDamage'].astype(bool)
    df_test['hasDamage'] = df_test['hasDamage'].astype(bool)

    return X_train, X_val, df_test

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def missing_values_treatment (X_train, X_val, df_test):

    """ Treatment of missing values. """

    ####PREENCHER

    

    return X_train, X_val, df_test

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def typecasting (X_train, X_val, df_test):

    """ Typecasting of variables (after filling in the missing values). """
    
    X_train['year'] = X_train['year'].astype(int)
    X_val['year'] = X_val['year'].astype(int)
    df_test['year'] = df_test['year'].astype(int)

    X_train['previousOwners'] = X_train['previousOwners'].astype(int)
    X_val['previousOwners'] = X_val['previousOwners'].astype(int)
    df_test['previousOwners'] = df_test['previousOwners'].astype(int)
  

    return X_train, X_val, df_test

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def feature_engineering (X_train, X_val, df_test):

    """ Feature engineering. 
    
    - **carAge** : The age of the car. 
        - It contains the same information as the year, but its interpretation is more straightforward.
    - **AvgUsage** : Shows average usage.
        - Total mileage is not comparable between cars of different ages.
    - **carSegment**: Segment to which the car brand belongs (Premium, Semi-premium, Economy).
        - Premium --> 3
        - Semi-premium --> 2
        - Economy --> 1
     
     """
    #Car Age
    X_train['carAge'] = (2020 - X_train['year']).round(0).astype(int) #0 means it's from 2020
    X_val['carAge'] = (2020 - X_val['year']).round(0).astype(int)
    df_test['carAge'] = (2020 - df_test['year']).round(0).astype(int)

    # Average Usage
    X_train['AvgUsage'] = X_train['mileage'] / (X_train ['carAge'] +1) #+1 to avoid divisions by 0
    X_val['AvgUsage'] = X_val['mileage'] / (X_val ['carAge'] +1)
    df_test['AvgUsage'] = df_test['mileage'] / (df_test ['carAge'] +1)
    
    # Car Segment based on Brand
    df_temp = pd.concat([X_train, y_train], axis=1) 
    brand_price = df_temp.groupby('Brand_cleaned')['price'].mean().sort_values()
    economy_limit = brand_price.quantile(0.33)
    semi_premium_limit = brand_price.quantile(0.66)

    segment = {brand: (1 if price <= economy_limit else 2 if price <= semi_premium_limit else 3) for brand, price in brand_price.items()}

    X_train['carSegment'] = X_train['Brand_cleaned'].map(segment)
    X_val['carSegment'] = X_val['Brand_cleaned'].map(segment)
    df_test['carSegment'] = df_test['Brand_cleaned'].map(segment)

    return X_train, X_val, df_test

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def encoding (X_train, X_val, y_train, df_test):

    """ Encoding of categorical variables. """

    target_features = ['model_cleaned','Brand_cleaned']
    one_hot_features = ['fuelType_cleaned', 'transmission_cleaned']

    encoder = TargetEncoder(categories="auto" , target_type="continuous" ) 

    encoder.fit(X_train[target_features], y_train)

    X_train[[f'{feat}_encoded' for feat in target_features]] = encoder.transform(X_train[target_features])
    X_val[[f'{feat}_encoded' for feat in target_features]] = encoder.transform(X_val[target_features])
    df_test[[f'{feat}_encoded' for feat in target_features]] = encoder.transform(df_test[target_features])

    for feat in one_hot_features:
        encoder = OneHotEncoder(drop='first', categories="auto" , handle_unknown="ignore", sparse_output=False).set_output(transform='pandas')
        encoder.fit(X_train[[feat]])

        train = encoder.transform(X_train[[feat]])
        val = encoder.transform(X_val[[feat]])
        test = encoder.transform(df_test[[feat]])

        X_train = pd.concat ([X_train, train], axis=1)
        X_val = pd.concat ([X_val, val], axis=1)
        df_test =pd.concat ([df_test, test], axis=1)

    # Note: we coded carSegment (ordinal encoder) when the variable was created.
    
    return X_train, X_val, df_test

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def scaling (X_train, X_val, df_test, scaler= StandardScaler()):

    """ Scaling of numerical variables. """

    scaler = StandardScaler()

    numeric_features = ['year', 'mileage', 'tax', 'mpg', 'engineSize', 'paintQuality%',	'previousOwners', 'carAge', 'AvgUsage',
                    'model_cleaned_encoded', 'Brand_cleaned_encoded']

    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    df_test_scaled = df_test.copy()

    scaler.fit(X_train_scaled[numeric_features])

    X_train_scaled[numeric_features] = scaler.transform(X_train_scaled[numeric_features])
    X_val_scaled[numeric_features] = scaler.transform(X_val_scaled[numeric_features])
    df_test_scaled[numeric_features] = scaler.transform(df_test_scaled[numeric_features])

    return X_train_scaled, X_val_scaled, df_test_scaled

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def feature_selection (X_train, X_train_scaled, X_val_scaled, df_test_scaled):

    """ Feature selection based on Filter Methods, Wrapper Methods and Embendded Methods. """

    ## First, lets drop the variables without corrections and the uncoded categorical variables. 
    ## We'll keep all variables in X_train, X_val, and df_test because we might need to make some kind of comparison with the original data in the future. 
    ## We'll just remove variables from the already scaled datasets: X_train_scaled, X_val_scaled, df_test_scaled. 

    X_train_scaled.drop(['Brand', 'model', 'transmission', 'fuelType','Brand_cleaned', 'transmission_cleaned', 'fuelType_cleaned',
                     'model_cleaned'], axis=1, inplace=True)
    X_val_scaled.drop(['Brand', 'model', 'transmission', 'fuelType','Brand_cleaned', 'transmission_cleaned', 'fuelType_cleaned',
                     'model_cleaned'], axis=1, inplace=True)
    df_test_scaled.drop(['Brand', 'model', 'transmission', 'fuelType','Brand_cleaned', 'transmission_cleaned', 'fuelType_cleaned',
                     'model_cleaned'], axis=1, inplace=True)
    
    ## As for the one hot encoded variables, we choose to keep all the categories.

    to_choose= ['year', 'mileage', 'tax', 'mpg', 'engineSize', 'paintQuality%',
       'previousOwners', 'hasDamage', 'carAge', 'AvgUsage',
       'model_cleaned_encoded', 'Brand_cleaned_encoded', 'carSegment']
    
    X_train_aux = X_train_scaled[to_choose].copy()
    X_val_aux = X_val_scaled[to_choose].copy()
    df_test_aux = df_test_scaled[to_choose].copy()

    X_train_aux ['hasDamage_num'] = X_train_aux['hasDamage'].astype(int)

    summary = pd.DataFrame (columns =['Variance', 'Spearman', 'RFE LR', 'RFE Lasso', 'LASSO', 'Sum'], index=to_choose)

    ######----------------------------------------------------------------FILTER METHODS--------------------------------------------------------------------######

    ## Check for constant variables (variance = 0)
    variances = X_train[to_choose].var()
    if (variances == 0).all():
        summary = summary.drop(columns=['Variance'])
    
    ## Spearman correlation
    X_train_scaled ['hasDamage_num'] = X_train_scaled['hasDamage'].astype(int)
    
    ### Before computing the rest of the correlations: year and carAge are perfectly correlated, so we will drop year because carAge is easier to interpret.
    summary.loc['year', :] = 0

    #COMPLETARRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR

    ######----------------------------------------------------------------WRAPPER METHODS--------------------------------------------------------------------######

    


    
