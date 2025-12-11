# **Car's price prediction**

## **Context**
The company Cars 4 You resales cars online. They are struggling with increasing waiting lists since the cars need to be evaluated by a mechanic. This can become a problem that prevents the company from continuing to grow, as it may lead current clients and potential future customers to turn to competitors instead. The goal of this project is to create a regression model that is capable of predicting the price of the car only with the information provided by the client.

## **Metadata**
| Variable | Description |
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

  ## **Feature Selection**

Firstly, select only the features with variance above 0.01. Then 3 methods are applied
- Correlation between the features and the target
- Recursive Feature Selection (RFE) with Linear Regression
- Obtain Decision Tree Feature Importance

The feature will be selected if it is selected by at least 2 of these methods. 
The number of features selected is one of the parameters testes with the function RandomizedSearchCV
   
## **Model Selection and hyperparameter tuning**
| Model | Parameters | Test MAE | Overfit |
| Gradient Boosting |  |  |
| Neural Network |  |  |
| Random Forest |  |  |
| K Nearest Neighbors |  |  |
| Decision Tree |  |  |
| Huber Regressor |  |  |
| Linear Regression |  |  |

## **Ablation Study**

## **Feature Importance**


## **Predictions interface**
Users can submit a csv file with the car's information or they can manually insert that information. The interface uses the final model to calculate the predictions.
Link for the interface:
https://machine-learning-project-pvymzmh8w8nyml294qqvvc.streamlit.app/
