# ------------IMPORTS-----------------------------------------------------

import pandas as pd
import numpy as np

# Class implementation:
from sklearn.base import BaseEstimator, TransformerMixin


# Feature selection:
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import RFE

# Scaling:
from sklearn.preprocessing import TargetEncoder, StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler


# Helper functions:
from functions import *

import warnings
warnings.filterwarnings('ignore')



# All the classes defined in this script will inherit from the BaseEstimator and TransformerMixin sklearn classes:
    # BaseEstimator: allows for class integration in pipeline and RandomizedSearchCV
    # TransformerMixin: allows for the combination of fit and transform methods, improving pipeline efficiency

# All additional functions which we created and are called inside some of the classes are defined in the functions.py script.


# ---------------------------------------------------- CLASSES INDEX ---------------------------------------------------------

# 1. Categorial_Correction
# 2. Outlier_Treatment
# 3. Missing_Value_Treatment
# 4. Typecasting
# 5. Feature_Engineering
# 6. Encoder
# 7. Scaler
# 8. Feature_Selection
# 9. Identity_Transformer
# 10. Simplified_Categorical_Correction
# 11. Simplified_Missing_Value_Treatment
# 12. Simplified_Encode




# ------------ Categorical Variable Typo Correction --------------------------------------------------------------------

class Categorical_Correction(BaseEstimator, TransformerMixin): 
    """
    Transformer for cleaning typos in categorical features. It learns mappings for each of the features during the fit method (based only
    on train data), using clustering and similarity-based functions. The fuzzywuzzy based function, create_clusters(), is used to 
    build mappings for brand, fuelType and transmission. For model, similar_models(), which is based in get_close_matches from difflib,
    is applied. All the mappings are used to clean data in the transform method.

    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        

        X = X.copy()

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        ###### -----------------------------------------------------BRAND----------------------------------------------------------------------######

        self.brands_ = X['Brand'].dropna().unique().tolist()

        _, self.mapping_brand_ = create_clusters(X, self.brands_, 'Brand')

        ######---------------------------------------------------TRANSMISSION--------------------------------------------------------------------######

        self.transmission_types_ = X['transmission'].dropna().unique().tolist()

        _, self.mapping_transmission_ = create_clusters(X, self.transmission_types_, 'transmission')

        ######----------------------------------------------------FUEL TYPE--------------------------------------------------------------------######

        self.fuel_types_ = X['fuelType'].dropna().unique().tolist()

        _, self.mapping_fueltype_ = create_clusters(X, self.fuel_types_, 'fuelType')

        ######------------------------------------------------------MODEL----------------------------------------------------------------------######

        self.models_ = X['model'].dropna().unique().tolist()

        ## Fuzzywuzzy wasn't able to group the same models in the column 'model', so for this case we will use get_close_matches from the difflib library,
        ## which we incorporated into the similar_models() function (check functions.py).

        _, self.mapping_model_ = similar_models(X, self.models_)

        return self

    def transform(self, X):

        X = X.copy()

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # BRAND----------------------------------------------------------------------------------------------
        X['Brand_cleaned'] = X['Brand'].apply(lambda x: correct_categorical(self.mapping_brand_, x))

        # TRANSMISSION---------------------------------------------------------------------------------------

        X['transmission_cleaned'] = X['transmission'].apply(
            lambda x: correct_categorical(self.mapping_transmission_, x))

        # FUELTYPE ------------------------------------------------------------------------------------------
        X['fuelType_cleaned'] = X['fuelType'].apply(lambda x: correct_categorical(self.mapping_fueltype_, x))

        # MODEL ---------------------------------------------------------------------------------------------
        X['model_cleaned'] = X['model'].apply(
            lambda x: correct_column_model(x, self.mapping_model_))
        
        #After analyzing the data (EDA), we noticed that after corrections, one model was associated with two brands. We will manually fix this error:
        X.loc[(X['model_cleaned']=='I3') & (X['Brand_cleaned'] == 'HYUNDAI'), 'model_cleaned'] == 'I30'

        # Finally, we drop the variables with typos:
        X = X.drop(['Brand', 'model', 'transmission', 'fuelType'], axis=1)

        return X




# -------------------------- OUTLIER TREATMENT -----------------------------------------------------

class Outlier_Treatment(BaseEstimator, TransformerMixin):
    """
    Transformer used in numerical feature outlier treatment. It identifies outliers using the IQR method during fit, stores the
    lower and upper limits for each feature, and applies winsorization during the transform method if the proportion of outliers exceeds a 
    certain ratio, whose default is set to 0.00008 (0.008%).

    -----------------------------

    Parameter: 
    - ratio: minimum upper/lower outlier proportion in order to apply winsorization

    """

    def __init__(self, ratio=0.00008):
        self.feat_lst = ['tax', 'mileage', 'mpg', 'engineSize', 'year', 'previousOwners'] # list of numerical feeatures we want to treat
        self.ratio = ratio # for winsorization (see below)

    def fit(self, X, y=None):

        X = X.copy()

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.outlier_limits_ = {} # dictionary to save outlier IQR limits for each feature
        self.quantiles_ = {} # dictionary to save 1st and 99th quantiles for eeach feature

        for feat in self.feat_lst:
            Q1 = X[feat].quantile(0.25)
            Q3 = X[feat].quantile(0.75)
            IQR = Q3 - Q1
            self.outlier_limits_[feat] = (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR) # saving limits as tuple
            self.quantiles_[feat] = (X[feat].quantile(0.01), X[feat].quantile(0.99)) # saving quantiles as tuple

        return self

    def transform(self, X):

        X = X.copy()

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        for feat in self.feat_lst: # for each feature:

            upper = self.outlier_limits_[feat][1] # retrieving upper IQR outlier limit
            lower = self.outlier_limits_[feat][0] # retrieving lower IQR outlier limit

            # if the ratio of upper outliers is over the class defined ratio, we winsorize at upper limit:
            if X[X[feat] > upper].shape[0] / X.shape[0] >= self.ratio:
                X.loc[X[feat] > upper, feat] = self.quantiles_[feat][1]
            # if the ratio of lower outliers is over the class defined ratio, we winsorize at lower limit:
            if X[X[feat] < lower].shape[0] / X.shape[0] >= self.ratio:
                X.loc[X[feat] < lower, feat] = self.quantiles_[feat][0]

        return X



# -------------------- MISSING VALUES TREATMENT -----------------------------------------------------

class Missing_Value_Treatment(BaseEstimator, TransformerMixin):
    """
    Transformer for imputation of missing values across numerical and categorical features. In the fit method, 
    it learns either the most frequent value or a feature-dependent mapping. The transform method applies these 
    learned mappings to impute the missing values in each feature.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):

        X = X.copy()

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)


        # Missing BRAND -----------------------------------------------------------------------------------------------
        # the global mode and model_to_brand mapping will be plugged into impute_brand function in the transform method
        self.brand_mode_ = X['Brand_cleaned'].mode().iloc[0] # learning the global mode
        self.model_to_brand_ = (X.dropna(subset=['Brand_cleaned', 'model_cleaned'])
                            .groupby('model_cleaned')['Brand_cleaned']
                            .agg(lambda x: x.mode().iloc[0])  # learning the most frequent brand for each model
                            .to_dict())
        

        # Missing MODEL -----------------------------------------------------------------------------------------------
        # we build a mapping through build_model_mapping(); this mapping is passed into impute_model_flexible in transform
        self.model_maps_ = build_model_mappings(X)


        # Missing YEAR ------------------------------------------------------------------------------------------------
        # we create bins from dividing mileage into quantiles:
        self.bins_ = X['mileage'].quantile([0, 0.2, 0.4, 0.6, 0.8, 1]).values
        # bin class labels:
        self.labels_ = ['very low', 'low', 'average', 'high', 'very high']
        # we create a mileage_bin column which classifies each data point into one mileage class
        X['mileage_bin'] = pd.cut(X['mileage'], bins=self.bins_, labels=self.labels_, include_lowest=True)

        # we get the median of year:
        self.year_median_ = X['year'].median()

        # we build the map using only build_year_mappings()
        self.year_maps_ = build_year_mappings(X)


        # Missing MILEAGE ---------------------------------------------------------------------------------
        # we group by year and get the median of mileage to build the map
        self.mileage_map_ = (X.dropna(subset=['mileage','year']).groupby(['year'])['mileage'].median().to_dict())


        # Missing TAX -------------------------------------------------------------------------------------
        # we build the map using only build_tax_mappings()
        self.tax_maps_ = build_tax_mappings(X)


        # Missing FUELTYPE --------------------------------------------------------------------------------
        # we build the map using only build_fuel_mappings()
        self.fueltype_maps_ = build_fuel_mappings(X)


        # Missing MPG -------------------------------------------------------------------------------------
        # we build the map using only build_mpg_mappings()
        self.mpg_maps_ = build_mpg_mappings(X)


        # Missing ENGINE SIZE -----------------------------------------------------------------------------
        # we build the map using only build_engine_mappings()
        self.enginesize_maps_ = build_engine_mappings(X)


        # Missing PREVIOUS OWNERS -------------------------------------------------------------------------
        # we build the map using only build_owners_mappings()
        self.previous_owners_maps_ = build_owners_mappings(X)


        # Missing TRANSMISSION ----------------------------------------------------------------------------
        # we build the map using only build_transmission_mappings()
        self.transmission_maps_ = build_transmission_mappings(X)

        # Missing HASDAMAGE -------------------------------------------------------------------------------
        # We just replace NaN values with True in transform()

        return self

    def transform(self, X):
        # this method combines the mappings created in fit with the functions that use them to impute
        # the missing values

        X = X.copy()

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if 'mileage_bin' in X.columns: # drop previously created mileage_bin (present in train sets, not in validation sets)
            X = X.drop('mileage_bin', axis=1)

        # Missing BRAND --------------------------------------------------------------------------------
        X['Brand_cleaned'] = X.apply(lambda row: impute_brand(row, self.model_to_brand_, self.brand_mode_),axis=1)


        # Missing MODEL --------------------------------------------------------------------------------
        X['model_cleaned'] = X.apply(lambda row: impute_model_flexible(row, self.model_maps_),axis=1)


        # Missing YEAR ---------------------------------------------------------------------------------
        # mileage_bin column is built again using the bins and labels learned in fit
        X['mileage_bin'] = pd.cut(X['mileage'], bins=self.bins_, labels=self.labels_, include_lowest=True)
        # year is imputed according to the impute_year function which uses all the mapping options as parameters
        X['year'] = X.apply(lambda row: impute_year(row, self.year_maps_, self.year_median_),axis=1)

        X = X.drop('mileage_bin', axis= 1) # mileage_bin is dropped, we won't need need it any further


        # Missing MILEAGE ------------------------------------------------------------------------------
        X['mileage'] = X.apply(lambda row: impute_mileage(row, self.mileage_map_), axis=1)


        # Missing TAX ----------------------------------------------------------------------------------
        X['tax'] = X.apply(lambda row: impute_tax(row, self.tax_maps_), axis=1)

        # Missing FUELTYPE -----------------------------------------------------------------------------
        X['fuelType_cleaned'] = X.apply(lambda row: impute_fueltype(row, self.fueltype_maps_),axis=1)


        # Missing MPG ----------------------------------------------------------------------------------
        X['mpg'] = X.apply(lambda row: impute_mpg(row, self.mpg_maps_), axis=1)


        # Missing ENGINESIZE ---------------------------------------------------------------------------
        X['engineSize'] = X.apply(lambda row: impute_engine(row, self.enginesize_maps_), axis=1)


        # Missing PREVIOUS OWNERS ----------------------------------------------------------------------
        X['previousOwners'] = X.apply(lambda row: impute_owners(row, self.previous_owners_maps_), axis=1)


        # Missing TRANSMISSION -------------------------------------------------------------------------
        X['transmission_cleaned'] = X.apply(lambda row: impute_transmission(row, self.transmission_maps_),axis=1)


        # Missing HASDAMAGE --------------------------------------------------------------------------------
        X['hasDamage'] = X['hasDamage'].fillna(True)

        return X




# ------------------------- TYPECASTING  -------------------------------------------------------------------

class Typecasting(BaseEstimator, TransformerMixin):
    """
    This transformer typecasts features into the correct type. 'year' and 'previousOwners' are converted to int 
    and 'hasDamage' to bool.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        X = X.copy()

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return self

    def transform(self, X):
        X = X.copy()
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X['year'] = X['year'].astype(int)
        X['previousOwners'] = X['previousOwners'].astype(int)
        X['hasDamage'] = X['hasDamage'].astype(bool)

        return X




# ------------------- FEATURE ENGINEERING -----------------------------------------------------------------

class Feature_Engineering(BaseEstimator, TransformerMixin):
    """
    This transformer is responsible for the engineering of new useful features. The new features include:
    - Car age ('carAge'): calculated through the difference between the reference year (2020) and the year of the car;
    - Average car usage ('AvgUsage'): calculated as mileage divided by car age plus one to avoid division by zero;
    - Car segment ('carSegment'): based on the mean price of the brand, divided into three categories: economy (1), 
    semi-premium (2), and premium (3).
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # creating temporary dataframe:
        df_temp = X.copy()
        # adding the price column:
        df_temp['price'] = y.values
        # grouping by brand and getting the mean price of each brand:
        brand_price = df_temp.groupby('Brand_cleaned')['price'].mean().sort_values()
        # getting 33th and 66th quantile:
        economy_limit = brand_price.quantile(0.33)
        semi_premium_limit = brand_price.quantile(0.66)
        # bulding a car segment map:
        self.segment_ = {brand: (1 if price <= economy_limit else 2 if price <= semi_premium_limit else 3) for
                         brand, price in brand_price.items()}

        return self

    def transform(self, X):
        X = X.copy()

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Car Age
        X['carAge'] = (2020 - X['year']).round(0).astype(int)

        # Average Car Usage
        X['AvgUsage'] = X['mileage'] / (X['carAge'] + 1)

        # Car Segment based on Brand
        X['carSegment'] = X['Brand_cleaned'].map(self.segment_).fillna(2)

        return X



# ------------ ENCODING ----------------------------------------------------------------------------------

class Encoder(BaseEstimator, TransformerMixin):

    """
    This transformer encodes categorical features. 'model_cleaned' and 'Brand_cleaned' are encoded through
    target encoding due to their large number of categories. 'fuelType_cleaned' and 'transmission_cleaned' 
    are encoded through one-hot encoding. 
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):

        X = X.copy()

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # features encoded by target encoding:
        self.target_features_ = ['model_cleaned','Brand_cleaned']

        # creating the target encoder instance and fitting it:
        self.encoder_ = TargetEncoder(categories="auto" , target_type="continuous" )
        self.encoder_.fit(X[self.target_features_], y)

        # features encoded by one-hot encoding:
        self.one_hot_features_ = ['fuelType_cleaned', 'transmission_cleaned']

        # creating encoder instances and fitting them:
        self.encoder_fueltype_ = OneHotEncoder(categories="auto" , handle_unknown="ignore", sparse_output=False).set_output(transform='pandas')
        self.encoder_fueltype_.fit(X[['fuelType_cleaned']])

        self.encoder_transmission_ = OneHotEncoder(categories="auto" , handle_unknown="ignore", sparse_output=False).set_output(transform='pandas')
        self.encoder_transmission_.fit(X[['transmission_cleaned']])

        return self

    def transform(self, X):
        X = X.copy()

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # target encoded features
        X[[f'{feat}_encoded' for feat in self.target_features_]] = self.encoder_.transform(X[self.target_features_])

        # one_hot encoded dummy features
        dummies_fueltype = self.encoder_fueltype_.transform(X[['fuelType_cleaned']])
        dummies_transmission = self.encoder_transmission_.transform(X[['transmission_cleaned']])

        # adding encoded features to dataframe
        X = pd.concat([X, dummies_fueltype, dummies_transmission], axis=1)

        # dropping the non-encoded corresponding features
        X = X.drop(['Brand_cleaned', 'transmission_cleaned', 'fuelType_cleaned','model_cleaned'], axis=1)

        return X






# ------------ SCALLING ----------------------------------------------------------------------------------

class Scaler(BaseEstimator, TransformerMixin):

    """
    This transformer scales the features using a specified scaler, set to StandardScaler() as default. 
    It selects all numerical features in the dataset, except the ones that were one-hot encoded, and 
    scales them.

     -----------------------------

    Parameter: 
    - scaler: type of scaler used 

    """

    def __init__(self, scaler=None):
        if scaler is None:
            self.scaler = StandardScaler() # initialization with StandardScaler as default
        else:
            self.scaler = scaler
        
        self.feats_names_ = None

    def fit(self, X, y=None):

        X = X.copy()

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # selecting numerical features
        self.numeric_features_ = X.select_dtypes(include=[np.number]).columns.tolist()

        # excluding fuelType and transmission, which were OH encoded
        for feat in [c for c in self.numeric_features_ if c.startswith("fuelType")]:
            if feat in self.numeric_features_:
                self.numeric_features_.remove(feat)

        for feat in [c for c in self.numeric_features_ if c.startswith("transmission")]:
            if feat in self.numeric_features_:
                self.numeric_features_.remove(feat)

        # fitting the scaler to the selected features
        self.scaler.fit(X[self.numeric_features_])

        self.feats_names_ = X.columns.tolist()

        return self

    def transform(self, X):

        X = X.copy()

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # transforming features using the previously fitted scaler
        X[self.numeric_features_] = self.scaler.transform(X[self.numeric_features_])

        return X






# ------------ FEATURE SELECTION -----------------------------------------------------

class Feature_Selection(BaseEstimator, TransformerMixin):
    """
    This transformer performs feature selection combining filter, wrapper and embedded methods. The fit method
    starts by selecting the features with variance above a speficied variance threshold (default is 0.01). Spearman 
    correlation with the target is checked for the selected features. Only the features with correlation above a 
    speficied Spearman correlation threshold (default is 0.2) are kept. Then RFE with Linear Regression is performed.
    The number of features selected by RFE shall be specified but is set to 10 as default. Finally, a decision tree
    is deployed as a wrapper method and features are selected through feature importance. The final selected features
    consist of those selected by at least two of the selection methods.

    -----------------------------

    Parameters: 
    - rfe_k: number of features to be selected by RFE;
    - spearman_thr: Spearman correlation threshold;
    - var_threshold: variance threshold.

    """

    def __init__(self, rfe_k= 10, spearman_thr= 0.2, var_threshold=0.01):

        self.rfe_k = rfe_k
        self.spearman_thr = spearman_thr
        self.var_threshold = var_threshold
        self.selected_features_ = None

    def fit(self, X, y=None):

        X = X.copy()

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X['hasDamage'] = X['hasDamage'].astype(int)

        # From numerical features we keep only the ones with variance != 0 (non-constant)
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()

        # we only keep features with variance above threshold
        features = [f for f in numeric_features if X[f].var() > self.var_threshold]

        X_features = X[features]

        # Filter method: Spearman correlation with target
        corr_with_target = X_features.apply(lambda x: x.corr(y, method='spearman'))
        selected_spearman = corr_with_target[ abs(corr_with_target) > self.spearman_thr].index.tolist()  # each index is a feature

        # Wrapper method: RFE with linear regression
        model = LinearRegression()
        rfe = RFE(estimator=model, n_features_to_select=self.rfe_k)
        rfe.fit(X=X[features], y=y)
        rfe_features = pd.Series(rfe.support_, index=features)
        rfe_features_list = rfe_features[rfe_features].index.tolist()  # only chooses the features where RFE selected True


        # Embedded method: Decision Tree feature importances
        dt_importances = DecisionTreeRegressor().fit(X[features], y).feature_importances_
        mean_importance = 1 / len(features)
        selected_indeces = np.where(dt_importances > mean_importance)[0]
        selected_dt = list(np.array(features)[selected_indeces])


        # Combine results from all methods. We choose to keep a feature if it is selected by at least two of the three methods.
        feature_counts = {}
        for selected_features in [selected_spearman, rfe_features_list, selected_dt]:
                for feature in selected_features:
                    feature_counts[feature] = feature_counts.get(feature, 0) + 1

        
        final_features = [f for f, count in feature_counts.items() if count >= 2]

        self.selected_features_ = final_features

        return self

    def transform(self, X):

        X = X.copy()

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        selected_features = self.selected_features_

        # filtering only features selected in the fit method
        X = X[selected_features]

        return X


# ------------ IDENTITY TRANSFORMER -----------------------------------------------------

class Identity_Transformer (BaseEstimator, TransformerMixin):

    """
    This transformer performs no transformation on the data, it simply returns the input features unchanged.
    By replacing a transformer with the Identity_Transformer, the pipeline can move forward without applying that specific step.
    In practice, this behaves as if the corresponding step had been removed from the pipeline, while preserving compatibility with ablation studies.

    -----------------------------

    Parameters:
    - None
    """

    def __init__(self):
        pass
    
    def fit (self, X, y=None):
        return self
    
    def transform (self, X):
        X = X.copy()

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        return X

# ------------ SIMPLIFIED CATEGORICAL CORRECTION -----------------------------------------------------

class Simplified_Categorical_Correction (BaseEstimator, TransformerMixin):

    """
    This transformer creates *_cleaned categorical columns without applying any cleaning rules.
    This transformer is used to avoid pipeline errors when the categorical cleaning step is skipped.

    -----------------------------

    Parameters:
    - None
    """

    def __init__ (self):
        self.categorical_cols = ['Brand', 'model', 'fuelType', 'transmission']    
          
    def fit (self, X, y=None):
        X = X.copy()

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.categorical_cols_ = [col for col in self.categorical_cols if col in X.columns]
        
        return self



    def transform (self, X):   
        X = X.copy()

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        for col in self.categorical_cols_:
            X[col + "_cleaned"] = X[col]


        return X
    
#------------ SIMPLIFIED MISSING VALUE TREATMENT -----------------------------------------------------

class Simplified_Missing_Value_Treatment (BaseEstimator, TransformerMixin):

    """
    This transformer performs a simplified missing value treatment for both numerical and categorical variables.
         -For numerical features, missing values are replaced with the mean of the respective feature.
        -For categorical features, missing values are replaced with the mode of the respective feature.
        -Exception: the variable 'hasDamage' is treated separately and missing values are replaced with True during the transform stage.

    -----------------------------

    Parameters:
    - None
    """


    def __init__ (self):
        pass
    
    def fit (self, X, y=None):
        X = X.copy()

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        numerical = X.select_dtypes(include=[np.number]).columns.to_list()
        categorical = X.select_dtypes(exclude=[np.number]).columns.to_list()

        self.means_ = {}
        for feat in numerical:
            self.means_ [feat] = X[feat].mean(skipna=True)

        self.modes_ = {}
        for feat in categorical:
            self.modes_ [feat] = X[feat].mode(dropna=True)[0] #[0] to select the most frequent value, we can have more than one mode


        return self
        
    def transform (self, X):

        X = X.copy()

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Missing BRAND --------------------------------------------------------------------------------
        X['Brand_cleaned'] = X['Brand_cleaned'].fillna(self.modes_['Brand_cleaned'])

        # Missing MODEL --------------------------------------------------------------------------------
        X['model_cleaned'] = X['model_cleaned'].fillna(self.modes_['model_cleaned'])

        # Missing YEAR --------------------------------------------------------------------------------
        X['year'] = X['year'].fillna(self.means_['year'])

        # Missing MILEAGE --------------------------------------------------------------------------------
        X['mileage'] = X['mileage'].fillna(self.means_['mileage'])

        # Missing TAX --------------------------------------------------------------------------------
        X['tax'] = X['tax'].fillna(self.means_['tax'])

        # Missing FUELTYPE --------------------------------------------------------------------------------
        X['fuelType_cleaned'] = X['fuelType_cleaned'].fillna(self.modes_['fuelType_cleaned'])

        # Missing MPG --------------------------------------------------------------------------------
        X['mpg'] = X['mpg'].fillna(self.means_['mpg'])

        # Missing ENGINESIZE --------------------------------------------------------------------------------
        X['engineSize'] = X['engineSize'].fillna(self.means_['engineSize'])

        # Missing PREVIOUS OWNERS --------------------------------------------------------------------------------
        X['previousOwners'] = X['previousOwners'].fillna(self.means_['previousOwners'])

        # Missing TRANSMISSION --------------------------------------------------------------------------------
        X['transmission_cleaned'] = X['transmission_cleaned'].fillna(self.modes_['transmission_cleaned'])

        # Missing HASDAMAGE --------------------------------------------------------------------------------
        X['hasDamage'] = X['hasDamage'].fillna(True)

        return X
    
#------------ SIMPLIFIED ENCODE -----------------------------------------------------

class Simplified_Encode (BaseEstimator, TransformerMixin):

    """
    This transformer performs a simplified one-hot encoding of categorical variables that have already been cleaned.

    -----------------------------

    Parameters:
    - None
    """
        
    def __init__ (self):
        pass

    def fit (self, X, y=None):
        X = X.copy()

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.categorical_ = ['model_cleaned','Brand_cleaned','fuelType_cleaned','transmission_cleaned']

        self.encoder_model_ = OneHotEncoder(categories="auto" , handle_unknown="ignore", sparse_output=False).set_output(transform='pandas')
        self.encoder_model_.fit(X[['model_cleaned']])

        self.encoder_brand_ = OneHotEncoder(categories="auto" , handle_unknown="ignore", sparse_output=False).set_output(transform='pandas')
        self.encoder_brand_.fit(X[['Brand_cleaned']])

        self.encoder_fueltype_ = OneHotEncoder(categories="auto" , handle_unknown="ignore", sparse_output=False).set_output(transform='pandas')
        self.encoder_fueltype_.fit(X[['fuelType_cleaned']])

        self.encoder_transmission_ = OneHotEncoder(categories="auto" , handle_unknown="ignore", sparse_output=False).set_output(transform='pandas')
        self.encoder_transmission_.fit(X[['transmission_cleaned']])

        return self
    
    def transform (self, X):
        X = X.copy()

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.dummies_model_ = self.encoder_model_.transform(X[['model_cleaned']])
        self.dummies_brand_ = self.encoder_brand_.transform(X[['Brand_cleaned']])
        self.dummies_fueltype_ = self.encoder_fueltype_.transform(X[['fuelType_cleaned']])
        self.dummies_transmission_ = self.encoder_transmission_.transform(X[['transmission_cleaned']])

        X = pd.concat([X, self.dummies_model_, self.dummies_brand_, self.dummies_fueltype_, self.dummies_transmission_], axis=1)

        X = X.drop(self.categorical_, axis=1)

        return X