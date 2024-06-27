from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor as xgbr




def train_mlpr_model(parameter_n_estimators,parameter_max_features,parameter_min_samples_split,parameter_min_samples_leaf,parameter_random_state,parameter_criterion,parameter_bootstrap,parameter_oob_score):

        model = ''
        # Fit the initial model
        # model.fit(X_train, y_train)


def train_knn_model(parameter_n_estimators,parameter_max_features,parameter_min_samples_split,parameter_min_samples_leaf,parameter_random_state,parameter_criterion,parameter_bootstrap,parameter_oob_score):
  
        model = ''

        # Fit the initial model
        # model.fit(X_train, y_train)

def train_rfm_model(parameter_n_estimators, parameter_max_features, parameter_min_samples_split, parameter_min_samples_leaf, parameter_random_state, parameter_criterion, parameter_bootstrap, parameter_oob_score, X_train, y_train):
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

    # Fit the initial model
    model.fit(X_train, y_train)
    return model

def train_svr_model(parameter_n_estimators,parameter_max_features,parameter_min_samples_split,parameter_min_samples_leaf,parameter_random_state,parameter_criterion,parameter_bootstrap,parameter_oob_score):

        model = ''

        # Fit the initial model
        # model.fit(X_train, y_train)

def train_xgb_model(parameter_n_estimators,parameter_max_features,parameter_min_samples_split,parameter_min_samples_leaf,parameter_random_state,parameter_criterion,parameter_bootstrap,parameter_oob_score):

        model = ''

        # Fit the initial model
        # model.fit(X_train, y_train)

      