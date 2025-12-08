import sys
sys.path.insert(0, '.')
from Classes import *
import joblib
import pandas as pd

model = joblib.load('modelo_mlp_teste.pkl')
print('Model type:', type(model))
print('\nModel structure:')
print(model)
if hasattr(model, 'steps'):
    print('\n\nPipeline steps:')
    for name, step in model.steps:
        print(f'  - {name}: {type(step).__name__}')
