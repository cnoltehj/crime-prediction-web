from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from modelTransformationResponse import gridsearchcvResponse

def param_grids_all_models():

        param_grids = {
                'ANN (MLPRegressor)': {
                        'hidden_layer_sizes': [(50, 50), (100,)] , 
                        'activation': ['tanh', 'relu'], 
                        'solver': ['adam', 'sgd']
                        },
                'KNN': {
                        'n_neighbors': [3, 5, 7] , 
                        'weights': ['uniform', 'distance']
                        },
                'RFM': {
                        'n_estimators': [1 , 5 ,10 ,50 ,100 ,200],
                        'max_features': ['auto', 'sqrt', 'log2'],
                        'min_samples_split': [1, 2, 5, 10],
                        'min_samples_leaf': [1, 10, 2, 1],
                        'random_state': [42],
                        'criterion': ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
                        'bootstrap': [True, False],
                        'oob_score': [True, False]
                        },
                'SVR': {
                        'kernel': ['linear', 'rbf'], 
                        'C': [1, 10], 
                        'epsilon': [0.1, 0.2]
                        },
                'XGBoost': {
                        'n_estimators': [100, 200],
                        'learning_rate': [0.01, 0.1],
                        'max_depth': [3,5,7],
                        'random_state': [42],
                        'min_child_weight': [1, 3],
                        'subsample': [0.8, 1.0],
                        'colsample_bytree': [0.8, 1.0]
                        }
                }
        
        return param_grids
        
def param_grids_knn_model(parameter_n_neighbors, parameter_weights):
  
        return {
                'hidden_layer_sizes': [parameter_n_neighbors] , 
                'activation': [parameter_weights], 
                'solver': ['adam', 'sgd']
                }

def param_grids_rfm_model(parameter_n_estimators, parameter_max_features, parameter_min_samples_split, parameter_min_samples_leaf, parameter_random_state, parameter_criterion, parameter_bootstrap, parameter_oob_score):
    
        return{
                'n_estimators': [parameter_n_estimators],
                'max_features': [parameter_max_features],
                'min_samples_split': [parameter_min_samples_split],
                'min_samples_leaf': [parameter_min_samples_leaf],
                'random_state': [parameter_random_state],
                'bootstrap': [parameter_bootstrap],
                'oob_score': [parameter_oob_score],
                'criterion': [parameter_criterion]
                }

def param_grids_svr_model(parameter_kernel, parameter_C, parameter_epsilon):
         
        return {
                'kernel' : [parameter_kernel],
                'C' : [parameter_C],
                'epsilon': [parameter_epsilon]
                }

def param_grids_xgb_model(parameter_n_estimators, parameter_learning_rate, parameter_max_depth, parameter_min_child_weight, parameter_subsample_bytree, parameter_random_state):

        return {
                'n_estimators' : [parameter_n_estimators],
                'learning_rate' : [parameter_learning_rate],
                'max_depth' : [parameter_max_depth],
                'min_child_weight' : [parameter_min_child_weight],
                'subsample_bytree' : [parameter_subsample_bytree],
                'random_state' : [parameter_random_state]
                }

def param_grids_ann_model(parameter_hidden_layer_size,parameter_activation,parameter_solver):
  
        return  {
                'hidden_layer_sizes': [parameter_hidden_layer_size],
                'activation' : [parameter_activation],
                'solver' : [parameter_solver]
                }

