from sklearn.model_selection import train_test_split , GridSearchCV



# # Example for SVR with GridSearchCV
# grid_search_svr = GridSearchCV(svr_pipeline, param_grid_svr, cv=5, scoring='neg_mean_squared_error')
# grid_search_svr.fit(X_train, y_train)

# # Example for KNN with GridSearchCV
# grid_search_knn = GridSearchCV(knn_pipeline, param_grid_knn, cv=5, scoring='neg_mean_squared_error')
# grid_search_knn.fit(X_train, y_train)

# # Example for Random Forest with GridSearchCV
# grid_search_rfm = GridSearchCV(rfm_pipeline, param_grid_rfm, cv=5, scoring='neg_mean_squared_error')
# grid_search_rfm.fit(X_train, y_train)

# # Example for MLPRegressor with GridSearchCV
# grid_search_mlp = GridSearchCV(mlp_pipeline, param_grid_mlp, cv=5, scoring='neg_mean_squared_error')
# grid_search_mlp.fit(X_train, y_train)

# # Example for XGBoost with GridSearchCV
# grid_search_xgb = GridSearchCV(xgb_pipeline, param_grid_xgb, cv=5, scoring='neg_mean_squared_error')
# grid_search_xgb.fit(X_train, y_train)



# Function to perform GridSearchCV and return the best model
# Function to handle a single model instance and param grid, rather than assuming mlp_model is a dictionary
def mlp_gridSearchCV(model, param_grid, X_train, y_train):
    scoring_value = 'neg_mean_squared_error'
    cv_value = 5

    grid_search_model = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring_value, cv=cv_value,error_score='raise')
    grid_search_model.fit(X_train, y_train)

    # Get the best model from GridSearchCV
    best_model = grid_search_model.best_estimator_

    # Return the best model
    return best_model


# Function to handle a mlp_model is a dictionary - multiple parameters
def List_mlp_gridSearchCV(mlp_model, param_grid_value, X_train, y_train):
    scoring_value = 'neg_mean_squared_error'
    cv_value = 5
    best_estimators = {}  # Initialize the dictionary to store best models

    for model_name, model in mlp_model.items():
        param_grid = param_grid_value[model_name]
        grid_search_model = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring_value, cv=cv_value)
        grid_search_model.fit(X_train, y_train)

        # Get the best model from GridSearchCV
        best_estimators[model_name] = grid_search_model.best_estimator_

    # Return the dictionary containing the best models
    return best_estimators  

