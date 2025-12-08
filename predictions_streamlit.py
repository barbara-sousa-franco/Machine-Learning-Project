import joblib
import streamlit as st
import pandas as pd
import numpy as np

# Import classes BEFORE loading the model (required for joblib deserialization)
# Even though we don't use them directly, they need to be in the namespace
from Classes import Categorical_Correction, Outlier_Treatment, Missing_Value_Treatment, Typecasting, Feature_Engineering, Encoder, Scaler, Feature_Selection

random_state = 42


# Carregar modelo
model = joblib.load("modelo_mlp_teste.pkl")


def preprocess_data(df):
    """
    Apply the same preprocessing steps done in the notebook before feeding to the pipeline.
    This handles invalid/impossible values by converting them to NaN.
    """
    df = df.copy()
    
    # Convert impossible values to NaN for numeric features
    # Year must be <= 2020
    df.loc[df['year'] > 2020, 'year'] = np.nan
    
    # Mileage must be >= 0
    df.loc[df['mileage'] < 0, 'mileage'] = np.nan
    
    # Tax must be >= 0
    df.loc[df['tax'] < 0, 'tax'] = np.nan
    
    # MPG must be > 0 and >= 8
    df.loc[df['mpg'] < 8, 'mpg'] = np.nan
    
    # Previous Owners must be >= 0
    df.loc[df['previousOwners'] < 0, 'previousOwners'] = np.nan
    
    # Engine Size must be > 0 and >= 1
    df.loc[df['engineSize'] < 1, 'engineSize'] = np.nan
    
    # Round year and previousOwners to whole numbers (only for non-NaN values)
    df['year'] = df['year'].apply(lambda x: np.floor(x) if pd.notna(x) else np.nan)
    df['previousOwners'] = df['previousOwners'].apply(lambda x: np.floor(x) if pd.notna(x) else np.nan)
    
    # Round numeric features to 2 decimal places (only for non-NaN values)
    for feat in ['mileage', 'tax', 'mpg', 'engineSize']:
        df[feat] = df[feat].apply(lambda x: round(x, 2) if pd.notna(x) else np.nan)
    
    # Preprocess categorical variables: strip spaces and uppercase (only for non-NaN values)
    categorical_cols = ['Brand', 'model', 'fuelType', 'transmission']
    for col in categorical_cols:
        if col in df.columns:
            # Only apply string operations to non-NaN values
            df[col] = df[col].apply(
                lambda x: str(x).strip().upper() if pd.notna(x) and x != '' else np.nan
            )
    
    return df

st.title("Predict your car's price")

method = st.sidebar.radio("How to insert the car information",
                          ["Load a csv file", "Manually write the information"])

necessary_columns = ['Brand', 'engineSize', 'model', 'transmission', 'mpg','year', 'tax', 'mileage', 'previousOwners', 'fuelType', 'hasDamage']


numeric_features = ['year', 'mileage', 'tax', 'mpg', 'engineSize', 'previousOwners', 'hasDamage']
cat_features = ['Brand', 'model', 'transmission', 'fuelType']

if method == "Load a csv file":

    st.subheader("Load a csv file")
    file = st.file_uploader("Upload a csv file", type="csv")
    if file is not None:
        df = pd.read_csv(file)

        received_columns = set(df.columns)

        # make sure that the file uploaded has the columns needed for the model
        if not set(necessary_columns).issubset(received_columns):
            st.error("The file does not contain the necessary columns")
            st.write("Missing columns:" + str(set(necessary_columns) - received_columns))
            st.stop()

        # in case the file has the needed columns and more, we only keep the necessary columns
        df = df[list(necessary_columns)]

        st.write("Data loaded:")
        st.dataframe(df)

        if st.button("Predict Price"):
            try:
                # Apply preprocessing: convert impossible values to NaN and standardize format
                df_processed = preprocess_data(df[list(necessary_columns)].copy())

                # predict using the complete pipeline
                pred = model.predict(df_processed)
                st.success("Predictions completed successfully!")
                
                st.write("Predictions:")
                df_results = df_processed.copy()
                df_results['Predicted_Price'] = pred
                st.dataframe(df_results)

                # possible to download results
                csv_full = df_results.to_csv(index=False)
                st.download_button(label = "Download Full Dataset with Predictions", data = csv_full, file_name = "dataset_predictions.csv", mime = "text/csv")

                last_column = df_results[['Predicted_Price']]
                csv_last = last_column.to_csv(index=False)
                st.download_button(label = "Download Only the Predictions", data = csv_last, file_name = "predictions.csv", mime = "text/csv")

            except Exception as e:
                st.error("An error occurred: " + str(e))
                


if method == "Manually write the information":
    st.subheader("Manually write the information")

    # empty, single-lined data frame with columns with the names of the excepted features
    observation = pd.DataFrame(columns = necessary_columns)

    feature_values = {}
    st.write("Insert the values for each feature")


    for feat in observation.columns:
        if feat in numeric_features:
            if feat == 'hasDamage':
                value = int(st.number_input("Has Damage:",min_value = 0, max_value = 1, step = 1))
            elif feat == 'previousOwners':
                value = int(st.number_input("Previous Owners:",min_value = 0, step = 1))
            elif feat == 'year':
                value = int(st.number_input("Year:",min_value = 1990, max_value = 2020, step = 1))
            elif feat == 'engineSize':
                value = float(st.number_input("Engine Size:",min_value = 1.00, step = 0.01, format="%.2f"))
            elif feat == 'mpg':
                value = float(st.number_input("MPG:",min_value = 8.00, step = 0.01, format="%.2f"))
            elif feat == 'mileage':
                value = float(st.number_input("Mileage:",min_value = 0.00, step = 0.01, format="%.2f"))
            elif feat == 'tax':
                value = float(st.number_input("Tax:",min_value = 0.00, step = 0.01, format="%.2f"))
            else:
                value = float(st.number_input(feat + ":", min_value = 0.00, step = 0.01, format = "%.2f"))
        else:
            # For categorical features, require input (don't allow empty)
            value = st.text_input(feat, value="")
            # Always process as string: strip spaces and uppercase
            # Only convert to nan if truly empty after stripping
            if value.strip() == "":
                value = np.nan
            else:
                value = value.strip().upper()

        feature_values[feat] = value


    if st.button("Predict Price"):
        try:
            df_observation = pd.DataFrame([feature_values])

            # Apply preprocessing before prediction
            df_observation = preprocess_data(df_observation)
            st.dataframe(df_observation)

            # predict price
            prediction = model.predict(df_observation)[0] # with [0] we select only the value, otherwise it would be an np.array just with the value

            st.success(f"Predicted price: ${prediction:.2f}")

        except Exception as e:
            st.error("An error occurred: " + str(e))
            
