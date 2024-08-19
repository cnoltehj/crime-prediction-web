from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

def hyperparameter_knn_model(parameter_n_neighbors, parameter_weights) :
  
        model = KNeighborsRegressor(
                n_neighbors = parameter_n_neighbors, 
                weights = parameter_weights
        )

        return model

def hyperparameter_rfm_model(parameter_n_estimators, parameter_max_features, parameter_min_samples_split, parameter_min_samples_leaf, parameter_random_state, parameter_criterion, parameter_bootstrap, parameter_oob_score):
    
        model = RandomForestRegressor(
                n_estimators=parameter_n_estimators,
                max_features=parameter_max_features,
                min_samples_split=parameter_min_samples_split,
                min_samples_leaf=parameter_min_samples_leaf,
                random_state=parameter_random_state,
                criterion=parameter_criterion,
                bootstrap=parameter_bootstrap,
                oob_score=parameter_oob_score
                )

        return model

def hyperparameter_svr_model(parameter_kernel, parameter_C, parameter_epsilon) :

        model = SVR(
                kernel = parameter_kernel, 
                C = parameter_C, 
                epsilon = parameter_epsilon
                )

        return model

def hyperparameter_xgb_model(parameter_n_estimators, parameter_learning_rate, parameter_max_depth, parameter_min_child_weight, parameter_subsample_bytree, parameter_random_state) :

        model = XGBRegressor(
                n_estimators=parameter_n_estimators,
                random_state=parameter_random_state,
                learning_rate = parameter_learning_rate, 
                max_depth = parameter_max_depth, 
                min_child_weight = parameter_min_child_weight, 
                subsample_bytree = parameter_subsample_bytree,
                feature_name_in = ''
                )
        return model

def hyperparameter_ann_model(parameter_hidden_layer_size,parameter_activation,parameter_solver) :
  
        model = MLPRegressor(
                hidden_layer_sizes= parameter_hidden_layer_size,
                activation = parameter_activation,
                solver = parameter_solver
        )

        return model


