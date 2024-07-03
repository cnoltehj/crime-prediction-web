
from sklearn.model_selection import train_test_split , GridSearchCV


def mlp_gridSearchCV(mlp_model, param_grid_value, X_train, y_train):
    scoring_value = 'neg_mean_squared_error'
    cv_value = 3
    best_estimators = {}  # Initialize the dictionary to store best models

    for model_name, model in mlp_model.items():
        param_grid = param_grid_value[model_name]
        grid_search_model = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring_value, cv=cv_value)
        grid_search_model.fit(X_train, y_train)

        # Get the best model from GridSearchCV
        best_estimators[model_name] = grid_search_model.best_estimator_

    # Return the dictionary containing the best models
    return best_estimators  

