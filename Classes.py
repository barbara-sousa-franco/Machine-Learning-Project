# ------------IMPORTS-----------------------------------------------------
from sklearn.linear_model import LinearRegression, Lasso, HuberRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


from sklearn.feature_selection import RFE
from math import ceil
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, median_absolute_error, root_mean_squared_error


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import TargetEncoder, StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler



import warnings
warnings.filterwarnings('ignore')

from functions_MARISA import *

# ------------ CLASSES -----------------------------------------------------
# ------------ Categorical Correction -----------------------------------------------------
class Categorical_Correction(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """ Learns parameters from train data to then use in categorical variable correction inside
        transformer function of this same class."""

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

        ######------------------------------------------------------MODELS----------------------------------------------------------------------######

        self.models_ = X['model'].dropna().unique().tolist()

        ## Fuzzywuzzy wasn't able to group the same models in the column 'model', so for this case we will use get_close_matches from the difflib library.
        self.clusters_ = similar_models(self.models_)

        # Calculate counts once before the function
        self.model_counts_ = X['model'].value_counts().to_dict()

        # List to store the most frequent model in each cluster which will be considered the correct one
        self.correct_models_ = []

        #   Dictionary to map each model to its correct version
        self.model_mapping_ = {}

        # For loop to go over all the clusters
        for group in self.clusters_:

            # The best model of each cluster will be the one with the highest count in the train set, it will be added to the correct_models list
            best = max(group, key=lambda x: self.model_counts_.get(x, 0))
            self.correct_models_.append(best)

            # Map all models in the group to the best model
            for model in group:
                self.model_mapping_[model] = best

        return self

    def transform(self, X):

        X = X.copy()

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # BRAND----------------------------------------------------------------------------------------------

        # We first correct the 'W' brands which should be either VW or BMW
        # this correction does not depent on params learned from train data:
        X['Brand'] = X.apply(lambda row: correct_brand_w(X, row['Brand'], row['model']), axis=1)
        # for cars with no model where correct_brand_w doesn't work:
        X.loc[X['Brand'] == 'W', 'Brand'] = 'VW'

        # Remaining typos:
        X['Brand_cleaned'] = X['Brand'].apply(lambda x: correct_categorical(self.mapping_brand_, x))

        # TRANSMISSION---------------------------------------------------------------------------------------
        X['transmission_cleaned'] = X['transmission'].apply(
            lambda x: correct_categorical(self.mapping_transmission_, x))

        # FUELTYPE -------------------------------------------------------------------------------------------
        X['fuelType_cleaned'] = X['fuelType'].apply(lambda x: correct_categorical(self.mapping_fueltype_, x))

        # MODEL ---------------------------------------------------------------------------------------------
        X['model_cleaned'] = X['model'].apply(
            lambda x: correct_column_model(self.correct_models_, self.model_mapping_, x, self.clusters_))

        # Finally, we drop the variables with typos:
        X = X.drop(['Brand', 'model', 'transmission', 'fuelType'], axis=1)

        return X









    # ------------ OUTLIER TREATMENT -----------------------------------------------------

class Outlier_Treatment(BaseEstimator, TransformerMixin):

    def __init__(self, dim=3, ratio=0.00008):
        self.feat_lst = ['tax', 'mileage', 'mpg', 'engineSize', 'year', 'previousOwners']
        self.dim = dim
        self.ratio = ratio

    def fit(self, X, y=None):

        X = X.copy()

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.outlier_limits_ = {}
        self.quantiles_ = {}

        for feat in self.feat_lst:
            Q1 = X[feat].quantile(0.25)
            Q3 = X[feat].quantile(0.75)
            IQR = Q3 - Q1
            self.outlier_limits_[feat] = (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
            self.quantiles_[feat] = (X[feat].quantile(0.01), X[feat].quantile(0.99))

        return self

    def transform(self, X):

        X = X.copy()

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        outlier_counts = pd.Series(0, index=X.index)

        for feat in self.feat_lst:

            upper = self.outlier_limits_[feat][1]
            lower = self.outlier_limits_[feat][0]

            # counting in how many feats each point is an outlier
            # outlier_counts += ((X[feat] < lower) | (X[feat] >  upper)).astype(int)

            if X[X[feat] > upper].shape[0] / X.shape[0] >= self.ratio:
                X.loc[X[feat] > upper, feat] = self.quantiles_[feat][1]

            if X[X[feat] < lower].shape[0] / X.shape[0] >= self.ratio:
                X.loc[X[feat] < lower, feat] = self.quantiles_[feat][0]

        return X








# ------------ MISSING VALUES TREATMENT -----------------------------------------------------


class Missing_Value_Treatment(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):

        X = X.copy()

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)


        # Missing BRAND ---------------------------------------------------------------------------------
        self.brand_mode_ = X['Brand_cleaned'].mode().iloc[0]
        self.model_to_brand_ = (X.dropna(subset=['Brand_cleaned', 'model_cleaned'])
                            .groupby('model_cleaned')['Brand_cleaned']
                            .agg(lambda x: x.mode().iloc[0])  # get most frequent brand for each model
                            .to_dict())

        # Missing MODEL ---------------------------------------------------------------------------------
        self.model_maps_ = build_model_mappings(X)

        # Missing YEAR ---------------------------------------------------------------------------------
        self.bins_ = X['mileage'].quantile([0, 0.2, 0.4, 0.6, 0.8, 1]).values
        self.labels_ = ['very low', 'low', 'average', 'high', 'very high']
        self.year_median_ = X['year'].median()

        X['mileage_bin'] = pd.cut(X['mileage'], bins=self.bins_, labels=self.labels_, include_lowest=True)

        self.year_map_mileage_ = (X.dropna(subset=['year', 'mileage_bin']).groupby('mileage_bin', observed=False)['year'].median().to_dict())

        self.year_map_tax_ = (X.dropna(subset=['year', 'tax']).groupby('tax', observed=False)['year'].median().to_dict())

        self.year_map_mpg_ = (X.dropna(subset=['year', 'mpg']).groupby('mpg', observed=False)['year'].median().to_dict())

        # Missing MILEAGE ---------------------------------------------------------------------------------
        self.mileage_map_ = (X.dropna(subset=['mileage','year']).groupby(['year'])['mileage'].median().to_dict())

        # Missing TAX --------------------------------------------------------------------------------
        self.tax_maps_ = build_tax_mappings(X)

        # Missing FUELTYPE --------------------------------------------------------------------------------
        self.fueltype_maps_ = build_fuel_mappings(X)

        # Missing MPG -----------------------------------------------------------------------------------------
        self.mpg_maps_ = build_mpg_mappings(X)

        # Missing ENGINE SIZE --------------------------------------------------------------------------------
        self.enginesize_maps_ = build_engine_mappings(X)


        # Missing PREVIOUS OWNERS --------------------------------------------------------------------------------
        self.previous_owners_maps_ = build_owners_mappings(X)

        # Missing TRANSMISSION --------------------------------------------------------------------------------
        self.transmission_maps_ = build_transmission_mappings(X)

        # Missing HASDAMAGE --------------------------------------------------------------------------------
        # We just replace NaN values with True in transform()

        return self

    def transform(self, X):
        X = X.copy()

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if 'mileage_bin' in X.columns:
            X = X.drop('mileage_bin', axis=1)

        # Missing BRAND --------------------------------------------------------------------------------
        X['Brand_cleaned'] = X.apply(lambda row: impute_brand(row, self.model_to_brand_, self.brand_mode_),axis=1)

        # Missing MODEL --------------------------------------------------------------------------------
        X['model_cleaned'] = X.apply(lambda row: impute_model_flexible(row, self.model_maps_),axis=1)

        # Missing YEAR --------------------------------------------------------------------------------
        X['mileage_bin'] = pd.cut(X['mileage'], bins=self.bins_, labels=self.labels_, include_lowest=True)

        X['year'] = X.apply(lambda row: impute_year(row, self.year_map_mileage_, self.year_map_tax_, self.year_map_mpg_, self.year_median_),axis=1)

        X = X.drop('mileage_bin', axis= 1)

        # Missing MILEAGE --------------------------------------------------------------------------------
        X['mileage'] = X.apply(lambda row: impute_mileage(row, self.mileage_map_), axis=1)

        # Missing TAX --------------------------------------------------------------------------------
        X['tax'] = X.apply(lambda row: impute_tax(row, self.tax_maps_), axis=1)

        # Missing FUELTYPE --------------------------------------------------------------------------------
        X['fuelType_cleaned'] = X.apply(lambda row: impute_fueltype(row, self.fueltype_maps_),axis=1)

        # Missing MPG --------------------------------------------------------------------------------
        X['mpg'] = X.apply(lambda row: impute_mpg(row, self.mpg_maps_), axis=1)

        # Missing ENGINESIZE --------------------------------------------------------------------------------
        X['engineSize'] = X.apply(lambda row: impute_engine(row, self.enginesize_maps_), axis=1)


        # Missing PREVIOUS OWNERS --------------------------------------------------------------------------------
        X['previousOwners'] = X.apply(lambda row: impute_owners(row, self.previous_owners_maps_), axis=1)

        # Missing TRANSMISSION --------------------------------------------------------------------------------
        X['transmission_cleaned'] = X.apply(lambda row: impute_transmission(row, self.transmission_maps_),axis=1)

        # Missing HASDAMAGE --------------------------------------------------------------------------------
        X['hasDamage'] = X['hasDamage'].fillna(True)

        return X









# ------------ TYPECASTING  -----------------------------------------------------

class Typecasting(BaseEstimator, TransformerMixin):

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










# ------------ FEATURE ENGINEERING -----------------------------------------------------

class Feature_Engineering(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        X = X.copy()

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        df_temp = X.copy()
        df_temp['price'] = y.values
        brand_price = df_temp.groupby('Brand_cleaned')['price'].mean().sort_values()
        economy_limit = brand_price.quantile(0.33)
        semi_premium_limit = brand_price.quantile(0.66)

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
        X['carSegment'] = X['Brand_cleaned'].map(self.segment_)

        return X











# ------------ ENCODING -----------------------------------------------------

class Encoder(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):

        X = X.copy()

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # features encoded by target encoding:
        self.target_features_ = ['model_cleaned','Brand_cleaned']

        self.encoder_ = TargetEncoder(categories="auto" , target_type="continuous" )
        self.encoder_.fit(X[self.target_features_], y)

        # features encoded by one-hot encoding:
        self.one_hot_features_ = ['fuelType_cleaned', 'transmission_cleaned']

        self.encoder_fueltype_ = OneHotEncoder(categories="auto" , handle_unknown="ignore", sparse_output=False).set_output(transform='pandas')
        self.encoder_fueltype_.fit(X[['fuelType_cleaned']])

        self.encoder_transmission_ = OneHotEncoder(categories="auto" , handle_unknown="ignore", sparse_output=False).set_output(transform='pandas')
        self.encoder_transmission_.fit(X[['transmission_cleaned']])

        return self

    def transform(self, X):
        X = X.copy()

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # target
        X[[f'{feat}_encoded' for feat in self.target_features_]] = self.encoder_.transform(X[self.target_features_])

        # one_hot
        self.dummies_fueltype_ = self.encoder_fueltype_.transform(X[['fuelType_cleaned']])
        self.dummies_transmission_ = self.encoder_transmission_.transform(X[['transmission_cleaned']])

        X = pd.concat([X, self.dummies_fueltype_, self.dummies_transmission_], axis=1)

        X = X.drop(['Brand_cleaned', 'transmission_cleaned', 'fuelType_cleaned','model_cleaned'], axis=1)

        return X















# ------------ SCALLING -----------------------------------------------------

class Scaler(BaseEstimator, TransformerMixin):

    def __init__(self, scaler=StandardScaler()):
        self.scaler = scaler

    def fit(self, X, y=None):

        X = X.copy()

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.numeric_features_ = X.select_dtypes(include=[np.number]).columns.tolist()

        for feat in [c for c in self.numeric_features_ if c.startswith("fuelType")]:
            if feat in self.numeric_features_:
                self.numeric_features_.remove(feat)

        for feat in [c for c in self.numeric_features_ if c.startswith("transmission")]:
            if feat in self.numeric_features_:
                self.numeric_features_.remove(feat)

        self.scaler.fit(X[self.numeric_features_])

        return self

    def transform(self, X):

        X = X.copy()

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X[self.numeric_features_] = self.scaler.transform(X[self.numeric_features_])

        return X










# ------------ FEATURE SELECTION -----------------------------------------------------

class Feature_Selection(BaseEstimator, TransformerMixin):

    def __init__(self, rfe_options=[1, 14], spearman_options=[0.2, 0.25, 0.3], var_threshold=0.01):

        self.rfe_options = rfe_options
        self.spearman_options = spearman_options
        self.var_threshold = var_threshold

    def fit(self, X, y=None):

        X = X.copy()

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X['hasDamage'] = X['hasDamage'].astype(int)

        # From numerical features we keep only the ones with variance != 0 (non-constant)
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()

        non_constant_features = [f for f in numeric_features if X[f].var() > 0]

        features = [f for f in non_constant_features if X[f].var() > self.var_threshold]

        dt_importances = DecisionTreeRegressor().fit(X[features], y).feature_importances_
        mean_importance = 1 / len(features)
        selected_indeces = np.where(dt_importances > mean_importance)[0]
        selected_dt = list(np.array(features)[selected_indeces])

        results = []  # to store all combinations and MAE
        mae_scores = []

        # Loop over all parameter combinations
        for n_feats in self.rfe_options:
            for spearman_thr in self.spearman_options:
                mae_scores = []

                # Compute absolute Spearman correlation with target
                features_fold = X[features]
                corr_with_target = features_fold.apply(lambda x: x.corr(y, method='spearman'))
                selected_spearman = corr_with_target[
                    abs(corr_with_target) > spearman_thr].index.tolist()  # each index is a feature

                # Wrapper method: RFE with Linear Regression
                model = LinearRegression()
                rfe_lr = RFE(estimator=model, n_features_to_select=n_feats)
                rfe_lr.fit(X=X[features], y=y)
                rfe_lr_features = pd.Series(rfe_lr.support_, index=features)
                rfe_lr_features_list = rfe_lr_features[
                    rfe_lr_features].index.tolist()  # only chooses the features where RFE selected True

                # Majority vote: keep features that appear in more than or at least half of the methods
                feature_counts = {}
                for method in [selected_spearman, rfe_lr_features_list, selected_dt]:
                    for f in method:
                        feature_counts[f] = feature_counts.get(f, 0) + 1
                n_methods = len([selected_spearman, rfe_lr_features_list, selected_dt])
                threshold = n_methods // 2 + n_methods % 2
                final_features = [f for f, count in feature_counts.items() if count >= threshold]

                # Evaluate performance with selected features
                model = LinearRegression()
                model.fit(X[final_features], y)
                y_pred = model.predict(X[final_features])
                mae = mean_absolute_error(y, y_pred)
                mae_scores.append(mae)
                results.append({'features': final_features, 'mae': mae})

        # Select combination with lowest MAE
        self.best_ = min(results, key=lambda x: x['mae'])

        return self

    def transform(self, X):

        X = X.copy()

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        selected_features = self.best_['features']

        X = X[selected_features]

        return X





