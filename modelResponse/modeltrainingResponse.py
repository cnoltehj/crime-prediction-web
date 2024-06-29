from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

def train_knn_model(parameter_n_estimators,parameter_max_features,parameter_min_samples_split,parameter_min_samples_leaf,parameter_random_state,parameter_criterion,parameter_bootstrap,parameter_oob_score):
  
        model = KNeighborsRegressor(
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

def train_rfm_model(parameter_n_estimators, parameter_max_features, parameter_min_samples_split, parameter_min_samples_leaf, parameter_random_state, parameter_criterion, parameter_bootstrap, parameter_oob_score):
    
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

def train_svr_model(parameter_n_estimators,parameter_max_features,parameter_min_samples_split,parameter_min_samples_leaf,parameter_random_state,parameter_criterion,parameter_bootstrap,parameter_oob_score):

        model = SVR(
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

def train_xgb_model(parameter_n_estimators,parameter_max_features,parameter_min_samples_split,parameter_min_samples_leaf,parameter_random_state,parameter_criterion,parameter_bootstrap,parameter_oob_score):

        model = XGBRegressor(
                n_estimators=parameter_n_estimators,
                max_features=parameter_max_features,
                min_samples_split=parameter_min_samples_split,
                min_samples_leaf=parameter_min_samples_leaf,
                random_state=parameter_random_state,
                criterion=parameter_criterion,
                bootstrap=parameter_bootstrap,
                oob_score=parameter_oob_score,
                feature_name_in = ''
                )

        return model

def train_ann_model(parameter_n_estimators,parameter_max_features,parameter_min_samples_split,parameter_min_samples_leaf,parameter_random_state,parameter_criterion,parameter_bootstrap,parameter_oob_score):
  
        model = MLPRegressor(
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

      