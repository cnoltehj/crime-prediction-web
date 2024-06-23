from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

def run_ann(parameter_max_features : str, parameter_split_size : int , parameter_random_state : int, X : int, y : int):
        
         # Proceed with splitting and model training
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100-parameter_split_size)/100, random_state=parameter_random_state)

     # Adjust the computation of max_features
        if parameter_max_features == 'all':
            parameter_max_features = None  # Use None for RandomForestRegressor to consider all features
            parameter_max_features_metric = X.shape[1]  # Number of features
        elif parameter_max_features == 'sqrt' or parameter_max_features == 'log2':
            parameter_max_features_metric = parameter_max_features  # Keep track of the metric used
        else:
            parameter_max_features_metric = int(parameter_max_features)  # Convert to integer if numeric