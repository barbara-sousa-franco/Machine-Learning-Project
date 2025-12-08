
# This python script contains the functions used to preprocess the data inside the K-fold.

# Import general functions
from functions_MARISA import *

# Import to divide the train set, encode and scale variables
from sklearn.preprocessing import TargetEncoder, StandardScaler, OneHotEncoder

# Import models and metrics used
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFE



# Categorical Variable Treatment


def categorical_treatment (X_train, df_test):

    """ Treatment of categorical variables with typos. """
    
    ######--------------------------------------------------------------------BRAND--------------------------------------------------------------------######

    # Correct the  'W' values in the columns 'Brand' by applying the function correct_brand_w to all elements in the column
    X_train['Brand'] = X_train.apply(lambda row: correct_brand_w(X_train, row['Brand'], row['model']),axis=1)
    df_test['Brand'] = df_test.apply(lambda row: correct_brand_w(df_test, row['Brand'], row['model']), axis = 1)

    ## The ones that were fixed were all 'VW' so we will assume the same for the cases with null values:
    X_train.loc[X_train['Brand'] =='W' ,'Brand'] = 'VW'

    df_test.loc[df_test['Brand'] =='W' ,'Brand'] = 'VW'

    brands = X_train['Brand'].dropna().unique().tolist()

    clusters, mapping = create_clusters(X_train, brands, 'Brand')

    X_train['Brand_cleaned'] = X_train['Brand'].apply(lambda x: correct_categorical(mapping, x))
    df_test['Brand_cleaned'] = df_test['Brand'].apply(lambda x: correct_categorical(mapping, x))

    ######-------------------------------------------------------------TRANSMISSION--------------------------------------------------------------------######

    transmission_types = X_train['transmission'].dropna().unique().tolist()

    clusters, mapping = create_clusters(X_train, transmission_types, 'transmission')

    X_train['transmission_cleaned'] = X_train['transmission'].apply(lambda x: correct_categorical(mapping, x))
    df_test['transmission_cleaned'] = df_test['transmission'].apply(lambda x: correct_categorical(mapping, x))

    ######---------------------------------------------------------------FUEL TYPES--------------------------------------------------------------------######

    fuel_types=X_train['fuelType'].dropna().unique().tolist()

    clusters, mapping = create_clusters(X_train, fuel_types, 'fuelType')

    X_train['fuelType_cleaned'] = X_train['fuelType'].apply(lambda x: correct_categorical(mapping, x))
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
    df_test['model_cleaned'] =  df_test['model'].apply(lambda x: correct_column_model(correct_models, model_mapping, x,clusters))

    return X_train, df_test



#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Outlier Tratment 


def outlier_removal(X_train, df_test, y_train, dim): 

    feat_lst = ['tax', 'mileage', 'mpg', 'engineSize', 'year', 'paintQuality%', 'previousOwners']


    # we only remove outliers from the train sets, we still call df_test and X_val for coherence in the final function
    X_train = X_train.copy()
    df_test = df_test.copy()

    # series to count in how many feats each car is an outlier
    outlier_counts_train = pd.Series(0, index=X_train.index)


    # looping over every feat
    for feat in feat_lst:
        Q1 = X_train[feat].quantile(0.25)
        Q3 = X_train[feat].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        # outlier counters
        outlier_counts_train += ((X_train[feat] < lower) | (X_train[feat] > upper)).astype(int)

    keep_train = outlier_counts_train < dim
    X_train = X_train.loc[keep_train]
    # remove rows in the target as well
    y_train = y_train.loc[keep_train]


    return X_train, df_test, y_train
        


def outlier_cap(X_train, df_test, ratio):

    feat_lst = ['tax', 'mileage', 'mpg', 'engineSize', 'year', 'paintQuality%', 'previousOwners']

    X_train = X_train.copy()
    df_test = df_test.copy()

    for feat in feat_lst:
        Q1 = X_train[feat].quantile(0.25)
        Q3 = X_train[feat].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        if X_train[X_train[feat] > upper].shape[0]/ X_train.shape[0] >= ratio:
            X_train.loc[X_train[feat] > upper, feat] = X_train[feat].quantile(0.99)
            df_test.loc[df_test[feat] > upper, feat] = X_train[feat].quantile(0.99)


        if X_train[X_train[feat] < lower].shape[0]/ X_train.shape[0] >= ratio:
            X_train.loc[X_train[feat] < lower, feat] = X_train[feat].quantile(0.01)
            df_test.loc[df_test[feat] < lower, feat] = X_train[feat].quantile(0.01)

    return X_train, df_test


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def missing_values_treatment (X_train, df_test):

    """ Treatment of missing values. """

    # Missing Brand: -----------------------------------------------------------------------------------------------

    brand_mode = X_train['Brand_cleaned'].mode().iloc[0]
    model_to_brand = (X_train.dropna(subset=['Brand_cleaned', 'model_cleaned'])
                        .groupby('model_cleaned')['Brand_cleaned']
                        .agg(lambda x: x.mode().iloc[0])  # get most frequent brand for each model
                        .to_dict())

    X_train['Brand_cleaned'] = X_train.apply(lambda row: impute_brand(row, model_to_brand, brand_mode), axis=1)
    df_test['Brand_cleaned'] = df_test.apply(lambda row: impute_brand(row, model_to_brand, brand_mode), axis=1)

    # Missing Model: -----------------------------------------------------------------------------------------------

    maps = build_model_mappings(X_train)
    X_train['model_cleaned'] = X_train.apply(lambda row: impute_model_flexible(row, maps), axis=1)
    df_test['model_cleaned'] = df_test.apply(lambda row: impute_model_flexible(row, maps), axis=1)

    # Missing Year: -----------------------------------------------------------------------------------------------

    bins = X_train['mileage'].quantile([0, 0.2, 0.4, 0.6, 0.8, 1]).values
    labels = ['very low', 'low', 'average', 'high', 'very high']

    X_train['mileage_bin'] = pd.cut(X_train['mileage'], bins=bins, labels=labels, include_lowest=True)
    df_test['mileage_bin'] = pd.cut(df_test['mileage'], bins=bins, labels=labels, include_lowest=True)

    year_median = X_train['year'].median()

    year_map_mileage = (X_train.dropna(subset=['year', 'mileage_bin']).groupby('mileage_bin')['year'].median().to_dict())
    year_map_tax = (X_train.dropna(subset=['year', 'tax']).groupby('tax')['year'].median().to_dict())
    year_map_mpg = (X_train.dropna(subset=['year', 'mpg']).groupby('mpg')['year'].median().to_dict())

    X_train['year'] = X_train.apply(lambda row: impute_year(row, year_map_mileage, year_map_tax, year_map_mpg, year_median), axis=1)
    df_test['year'] = df_test.apply(lambda row: impute_year(row, year_map_mileage, year_map_tax, year_map_mpg, year_median), axis=1)    

    X_train.drop('mileage_bin', axis=1, inplace=True)
    df_test.drop('mileage_bin', axis=1, inplace=True)


    # Missing Mileage: -----------------------------------------------------------------------------------------------

    mileage_map = (X_train.dropna(subset=['mileage','year']).groupby(['year'])['mileage'].median().to_dict())
    X_train['mileage'] = X_train.apply(lambda row: impute_mileage(row, mileage_map), axis=1)
    df_test['mileage'] = df_test.apply(lambda row: impute_mileage(row, mileage_map), axis=1)

    # Missing Tax: ---------------------------------------------------------------------------------------------------

    maps = build_tax_mappings(X_train)
    X_train['tax'] = X_train.apply(lambda row: impute_tax(row, maps), axis=1)
    df_test['tax'] = df_test.apply(lambda row: impute_tax(row, maps), axis=1)   

    # Missing fuelType: ----------------------------------------------------------------------------------------------

    maps = build_fuel_mappings(X_train)
    X_train['fuelType_cleaned'] = X_train.apply(lambda row: impute_fueltype(row, maps), axis=1)
    df_test['fuelType_cleaned'] = df_test.apply(lambda row: impute_fueltype(row, maps), axis=1)

    # Missing mpg: --------------------------------------------------------------------------------------------------

    maps = build_mpg_mappings(X_train)
    X_train['mpg'] = X_train.apply(lambda row: impute_mpg(row, maps), axis=1)
    df_test['mpg'] = df_test.apply(lambda row: impute_mpg(row, maps), axis=1)

   # Missing engineSize: ---------------------------------------------------------------------------------------------

    maps = build_engine_mappings(X_train)
    X_train['engineSize'] = X_train.apply(lambda row: impute_engine(row, maps), axis=1)
    df_test['engineSize'] = df_test.apply(lambda row: impute_engine(row, maps), axis=1)

    # Missing paintQuality: -----------------------------------------------------------------------------------------

    maps = build_paint_mappings(X_train)
    X_train['paintQuality%'] = X_train.apply(lambda row: impute_paint(row, maps), axis=1)
    df_test['paintQuality%'] = df_test.apply(lambda row: impute_paint(row, maps), axis=1)

    # Missing previousOwners: -----------------------------------------------------------------------------------------

    maps = build_owners_mappings(X_train)
    X_train['previousOwners'] = X_train.apply(lambda row: impute_owners(row, maps), axis=1)
    df_test['previousOwners'] = df_test.apply(lambda row: impute_owners(row, maps), axis=1)

    # Missing transmission: -----------------------------------------------------------------------------------------

    maps = build_transmission_mappings(X_train)
    X_train['transmission_cleaned'] = X_train.apply(lambda row: impute_transmission(row, maps), axis=1)
    df_test['transmission_cleaned'] = df_test.apply(lambda row: impute_transmission(row, maps), axis=1)


    # Missing hasDamage: -----------------------------------------------------------------------------------------

    X_train['hasDamage'].isna().replace(True, inplace=True)
    df_test['hasDamage'].isna().replace(True, inplace=True)


    return X_train, df_test

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def typecasting (X_train, df_test):

    """ Typecasting of variables (after filling in the missing values). """
    
    X_train['year'] = X_train['year'].astype(int)
    df_test['year'] = df_test['year'].astype(int)

    X_train['previousOwners'] = X_train['previousOwners'].astype(int)
    df_test['previousOwners'] = df_test['previousOwners'].astype(int)


    X_train['hasDamage'] = X_train['hasDamage'].astype(bool) 
    df_test['hasDamage'] = df_test['hasDamage'].astype(bool)
  

    return X_train, df_test

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def feature_engineering (X_train, df_test, y_train):

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
    df_test['carAge'] = (2020 - df_test['year']).round(0).astype(int)

    # Average Usage
    X_train['AvgUsage'] = X_train['mileage'] / (X_train ['carAge'] +1) #+1 to avoid divisions by 0
    df_test['AvgUsage'] = df_test['mileage'] / (df_test ['carAge'] +1)
    
    # Car Segment based on Brand
    df_temp = pd.concat([X_train, y_train], axis=1) 
    brand_price = df_temp.groupby('Brand_cleaned')['price'].mean().sort_values()
    economy_limit = brand_price.quantile(0.33)
    semi_premium_limit = brand_price.quantile(0.66)

    segment = {brand: (1 if price <= economy_limit else 2 if price <= semi_premium_limit else 3) for brand, price in brand_price.items()}

    X_train['carSegment'] = X_train['Brand_cleaned'].map(segment)
    df_test['carSegment'] = df_test['Brand_cleaned'].map(segment)

    return X_train, X_val, df_test

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def encoding (X_train, y_train, df_test):

    """ Encoding of categorical variables. """

    target_features = ['model_cleaned','Brand_cleaned']
    one_hot_features = ['fuelType_cleaned', 'transmission_cleaned']

    encoder = TargetEncoder(categories="auto" , target_type="continuous" ) 

    encoder.fit(X_train[target_features], y_train)

    X_train[[f'{feat}_encoded' for feat in target_features]] = encoder.transform(X_train[target_features])
    df_test[[f'{feat}_encoded' for feat in target_features]] = encoder.transform(df_test[target_features])

    for feat in one_hot_features:
        encoder = OneHotEncoder(categories="auto" , handle_unknown="ignore", sparse_output=False).set_output(transform='pandas')
        encoder.fit(X_train[[feat]])

        train = encoder.transform(X_train[[feat]])
        test = encoder.transform(df_test[[feat]])

        X_train = pd.concat ([X_train, train], axis=1)
        df_test =pd.concat ([df_test, test], axis=1)

    # Note: we coded carSegment (ordinal encoder) when the variable was created.
    
    return X_train, df_test

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def scaling (X_train,df_test, scaler= StandardScaler()):

    """ Scaling of numerical variables. """

    numeric_features = ['year', 'mileage', 'tax', 'mpg', 'engineSize', 'paintQuality%',	'previousOwners', 'carAge', 'AvgUsage',
                    'model_cleaned_encoded', 'Brand_cleaned_encoded']

    X_train_scaled = X_train.copy()
    df_test_scaled = df_test.copy()

    scaler.fit(X_train_scaled[numeric_features])

    X_train_scaled[numeric_features] = scaler.transform(X_train_scaled[numeric_features])
    df_test_scaled[numeric_features] = scaler.transform(df_test_scaled[numeric_features])

    return X_train_scaled, df_test_scaled

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def full_preprocessing(X_train, y_train, df_test, dim, ratio):

    X_train_prep, df_test_prep = categorical_treatment(X_train.copy(), df_test.copy())

    X_train_prep, df_test_prep, y_train_prep = outlier_removal(X_train_prep,df_test_prep, y_train.copy(), dim)

    X_train_prep, df_test_prep = outlier_cap(X_train_prep,df_test_prep, ratio)

    X_train_prep, df_test_prep = missing_values_treatment(X_train_prep,df_test_prep)

    X_train_prep, df_test_prep = typecasting(X_train_prep,df_test_prep)

    X_train_prep, df_test_prep = feature_engineering(X_train_prep,df_test_prep, y_train_prep)

    X_train_prep, df_test_prep = encoding(X_train_prep,y_train_prep,df_test_prep)

    X_train_scaled, df_test_scaled = scaling(X_train_prep,df_test_prep, scaler= StandardScaler())

    ## Lets drop the variables without corrections and the uncoded categorical variables. 
    ## We'll keep all variables in X_train, X_val, and df_test because we might need to make some kind of comparison with the original data in the future. 
    ## We'll just remove variables from the already scaled datasets: X_train_scaled, X_val_scaled, df_test_scaled.

    X_train_scaled.drop(['Brand', 'model', 'transmission', 'fuelType','Brand_cleaned', 'transmission_cleaned', 'fuelType_cleaned',
                     'model_cleaned'], axis=1, inplace=True)
    df_test_scaled.drop(['Brand', 'model', 'transmission', 'fuelType','Brand_cleaned', 'transmission_cleaned', 'fuelType_cleaned',
                     'model_cleaned'], axis=1, inplace=True)

    return (
        X_train_prep,   # before scaling
        df_test_prep,
        X_train_scaled, # after scaling
        df_test_scaled,
        y_train_prep
    )

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def feature_selection (X_train_scaled, y_train, rfe_options=[8, 10], spearman_options=[0.2, 0.25, 0.3] ,var_threshold=0.01, cv_inner=3):

    """ Feature selection combining:

        - Filter: variance and Spearman
        - Wrapper: RFE with Linear Regression and RFE with Lasso
        - Embedded: Lasso

    Returns the final decision ('final_features') based on majority.

    """
    
    # Inner CV for hyperparameter tuning (RFE n_features, Spearman threshold)
    kf_inner = KFold(n_splits=cv_inner, shuffle=True, random_state=42)

    # Keep all non-constant numeric features
    X_train_scaled ['hasDamage'] = X_train_scaled['hasDamage'].astype(int) #for practical purposes
    features = [f for f in X_train_scaled.select_dtypes(include=[np.number]).columns if X_train_scaled[f].var() > 0]

    results = []  # to store all combinations and MAE

    # # Loop over all hyperparameter combinations
    for n_feats in rfe_options:
        for spearman_thr in spearman_options:
            mae_scores = []

            # Inner CV loop
            for train_idx, val_idx in kf_inner.split(X_train_scaled):
                X_tr, X_val = X_train_scaled.iloc[train_idx], X_train_scaled.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                # Keep only numerical features with variance above threshold
                features = [f for f in features if X_tr[f].var() > var_threshold]

                # Compute absolute Spearman correlation with target
                features_fold = X_tr[features]
                corr_with_target = features_fold.apply(lambda x: x.corr(y_tr, method='spearman'))
                selected_spearman = corr_with_target[abs(corr_with_target) > spearman_thr].index.tolist() # each index is a feature
                
                # Wrapper method: RFE with Linear Regression
                model = LinearRegression()
                rfe_lr = RFE(estimator=model, n_features_to_select=n_feats)
                rfe_lr.fit(X= X_tr[features], y= y_tr)
                rfe_lr_features = pd.Series(rfe_lr.support_, index=features)
                rfe_lr_features_list = rfe_lr_features[rfe_lr_features].index.tolist() # only chooses the features were RFE selected True

                # Wrapper method: RFE with Lasso
                model = Lasso()
                rfe_l = RFE(estimator=model, n_features_to_select=n_feats)
                rfe_l.fit(X= X_tr[features], y= y_tr)
                rfe_l_features = pd.Series(rfe_l.support_, index=features)
                rfe_l_features_list = rfe_l_features[rfe_l_features].index.tolist()

                # Embedded method: Lasso
                lasso = LassoCV()
                lasso.fit(X_tr[features], y_tr)
                coef = pd.Series(lasso.coef_, index = features)
                lasso_features = coef[coef != 0].index.tolist()

                # Majority vote: keep features that appear in more than or at least half of the methods
                feature_counts = {}
                for method in [selected_spearman, rfe_lr_features_list, rfe_l_features_list, lasso_features]:
                    for f in method:
                        feature_counts[f] = feature_counts.get(f,0)+1
                n_methods = len([selected_spearman, rfe_lr_features_list, rfe_l_features_list, lasso_features])
                threshold = n_methods//2 + n_methods%2
                final_features = [f for f,count in feature_counts.items() if count >= threshold]

                # Evaluate performance with selected features
                model = LinearRegression()
                model.fit(X_tr[final_features], y_tr)
                y_pred = model.predict(X_val[final_features])
                mae_scores.append(mean_absolute_error(y_val, y_pred))

            mean_mae = np.mean(mae_scores)
            results.append({'features': final_features, 'mae': mean_mae})

    # Select combination with lowest MAE
    best = min(results, key=lambda x: x['mae'])

    return best
