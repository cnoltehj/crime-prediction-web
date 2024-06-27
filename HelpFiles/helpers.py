import pandas as pd
import numpy as np

def data_to_plot():

    datatoplot = pd.DataFrame({
        'Contact crime (Crimes against the person)': np.random.rand(100),
        'TRIO Crime': np.random.rand(100),
        'Contact-related crime': np.random.rand(100),
        'Property-related crime': np.random.rand(100),
        'Other serious crime': np.random.rand(100),
        '17 Community reported serious crime': np.random.rand(100),
        'Crime detected as a result of police action': np.random.rand(100)
        })
    return datatoplot

def specified_categories_data ():

# Assuming specified_categories are your target variables
    specifCategories = [
        'Contact crime (Crimes against the person)',
        'TRIO Crime',
        'Contact-related crime',
        'Property-related crime',
        'Other serious crime',
        '17 Community reported serious crime',
        'Crime detected as a result of police action'
    ]

    return specifCategories

def paramGrid():
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_features': ['auto', 'sqrt', 'log2'],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_depth': [None, 10, 20, 30]
        }
    return param_grid

