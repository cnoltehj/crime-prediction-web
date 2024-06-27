from sklearn.model_selection import train_test_split , GridSearchCV


def gridSeachSV(X_train, y_train,X_test,param_grid_data,model): 
    # Perform GridSearchCV for hyperparameter tuning
    grid_search_model = GridSearchCV(estimator= model, param_grid=param_grid_data, scoring='neg_mean_squared_error', cv=3)
    grid_search_model.fit(X_train, y_train)

    #Get the best model from GridSearchCV
    best_mlp_model = grid_search_model.best_estimator_

    return best_mlp_model
    