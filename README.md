# **Car's price prediction**

## **Context**
The company Cars 4 You resales cars online. They are struggling with increasing waiting lists since the cars need to be evaluated by a mechanic. This can become a problem that prevents the company from continuing to grow, as it may lead current clients and potential future customers to turn to competitors instead. The goal of this project is to create a regression model that is capable of predicting the price of the car only with the information provided by the client.

## **Metadata**
| Variable | Description |  
|------------------|-------------|  
| carID | An attribute that contains an identifier for each car |  
| Brand | The car’s main brand (e.g. Ford, Toyota) |  
| model | The car model |  
| year | The year of Registration of the Car |  
| transmission | Type of transmission of the car (e.g. Manual, Automatic, Semi-Automatic) |  
| mileage | The total reported distance travelled by the car (in miles) |  
| tax | The amount of road tax (in £) that, in 2020, was applicable to the car in question |  
| fuelType | Type of Fuel used by the car (Diesel, Petrol, Hybrid, Electric) |  
| mpg | Average Miles per Gallon |  
| engineSize | Size of Engine in liters (Cubic Decimeters) |  
| paintQuality% | The mechanic’s assessment of the cars’ overall paint quality and hull integrity (filled by the mechanic during evaluation) |  
| previousOwners | Number of previous registered owners of the vehicle |  
| hasDamage | Boolean marker filled by the seller at the time of registration stating whether the car is damaged or not |  
| price | The car’s price when purchased by Cars 4 You (in £) |  

## **Preprocessing**
**Categorical Correction:** Fix typos by using the fuzzywuzzy and difflib libraries  
**Outlier Treatment:** Winsorization at 1%, 99% or both  
**Missing Values Treatment:** Median or Mode imputation after grouping certain domains, different for every feature  
**Typecasting:** Typecast 'year' and 'previousOwners' to integers and 'hasDamage' to boolean  
**Feature Engineering:** 3 new features were added:    
- carAge : 2020 - 'year'
- AvgUsage : 'mileage' / ('carAge' + 1)
- carSegment : One of 3 price segments each with increasing average price.

**Encoding**: Use TargetEncoder() for Brand and model and OneHotEncoder() for fuelType and Transmission  
**Scalling**: Different scalling techniques will be applied
  

  ## **Feature Selection**

Firstly, select only the features with variance above 0.01. Then 3 methods are applied  
- Spearman correlation between the features and the target  
- Recursive Feature Selection (RFE) with Linear Regression  
- Decision Tree Feature Importance  

The feature will be selected if it is selected by at least 2 of these methods.    
The number of features selected is one of the parameters testes with the function RandomizedSearchCV  

   
## **Model Selection and hyperparameter tuning**
| Model | Parameters | Test MAE | Overfit (%) |  
|-------------------|---------|---------------|-------------|
| GradientBoosting   | True    | 1346.783091   | 6.689326    |
| RandomForest       | False   | 1355.735228   | 13.974068   |
| MLP_adam           | False   | 1426.602219   | 17.712090   |
| KNN                | True    | 1528.286795   | 7.612548    |
| Decision Tree      | False   | 1701.438878   | 15.540444   |
| Huber              | False   | 2518.077855   | 0.188716    |

The best model is Gradient Boosting, it achieves the best test score and has low percentage of overfit


## **Ablation Study**

In this section we analyse the imporatance of each step of the pipeline by removing or replacing it by a basic approach while maintaining the other steps constant.

| Step | Step Tested               | Test MAE | Execution Time (s) | Overfit MAE | Delta |
|------|---------------------------|----------|--------------------|-------------|--------|
| 0    | `Full Pipeline`           | 1344.344522 | 149.333922 | 1.066628 | 0.000000 |
| 1    | `categorical treatment`   | 1401.119986 | 164.941620 | 1.069398 | 56.775464 |
| 2    | `outlier treatment`       | 1431.386469 | 166.268776 | 1.059067 | 87.041948 |
| 3    | `missing value treatment` | 1397.294585 | 119.356688 | 1.075345 | 52.950064 |
| 4    | `typecasting`             | 1346.296980 | 212.374082 | 1.066778 | 1.952458 |
| 5    | `feature engineering`     | 1350.250382 | 234.684779 | 1.070611 | 5.905860 |
| 6    | `encoder`                 | 1773.838641 | 278.422422 | 1.061655 | 429.494119 |
| 7    | `scaler`                  | 1439.989006 | 206.957186 | 1.065673 | 95.644484 |
| 8    | `feature selection`       | 1350.983491 | 210.098535 | 1.078840 | 6.638969 |

Delta represents the difference betweeen the test score of the full pipeline and the one of the current step. Clearly the encoder step is the most important one, followed by the scaler and outlier treatment. All the preprocesing steps and feature selection are crucial since the best score achieved is with the full pipeline.


## **Feature Importance**
According to the Feature Importance Analysis, there were 5 features that have significant importance for the model performance. From the Feature Importance we were able to see that the feature importance obtained from SHAP values is similar to the one obtain by feature_importances_ attribute of the Gradient Boosting Regressor. The features that contribute the most for the predictions are:  model_cleaned_encoded, transmission_cleaned_MANUAL, year, carAge, engineSize and mileage, in descending order. The features selected during feature selection contain all the ones that have a bigger contribution on the predictions.

## **Predictions interface**
Users can submit a csv file with the car's information or they can manually insert that information. The interface uses the final model to calculate the predictions.  
Link for the interface:  
https://machine-learning-project-pvymzmh8w8nyml294qqvvc.streamlit.app/
