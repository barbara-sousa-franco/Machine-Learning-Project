import sys
sys.path.insert(0, '.')
from Classes import *
import joblib
import pandas as pd
import numpy as np

print("Loading model...")
model = joblib.load('modelo_mlp_teste.pkl')
print("✓ Model loaded successfully!")

print("\nTesting with example data...")

# Test case 1: CSV-like data
test_data_csv = pd.DataFrame({
    'Brand': ['VOLKSWAGEN'],
    'model': ['GOLF'],
    'year': [2004],
    'transmission': ['MANUAL'],
    'mileage': [170000.0],
    'fuelType': ['DIESEL'],
    'tax': [120.0],
    'mpg': [40.0],
    'engineSize': [4.5],
    'previousOwners': [2],
    'hasDamage': [0]
})

print("\nTest 1 - CSV Input:")
print(test_data_csv)
try:
    pred1 = model.predict(test_data_csv)
    print(f"✓ Prediction successful: ${pred1[0]:.2f}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test case 2: Manual input
test_data_manual = pd.DataFrame({
    'Brand': ['VOLKSWAGEN'],
    'model': ['GOLF'],
    'year': [2004],
    'transmission': ['MANUAL'],
    'mileage': [170000.0],
    'fuelType': ['DIESEL'],
    'tax': [120.0],
    'mpg': [40.0],
    'engineSize': [4.5],
    'previousOwners': [2],
    'hasDamage': [0]
})

print("\n\nTest 2 - Manual Input:")
print(test_data_manual)
try:
    pred2 = model.predict(test_data_manual)
    print(f"✓ Prediction successful: ${pred2[0]:.2f}")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n\nAll tests completed!")
