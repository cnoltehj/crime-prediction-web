
from sklearn.model_selection import train_test_split , GridSearchCV


def mse_gridSearchCV(mlp_model_value,param_grid_value,X_train,y_train,scroring_value = None, cv_value = None):
    scroring_value = 'neg_mean_squared_error'
    cv_value = 3
    grid_search_model = GridSearchCV(estimator=mlp_model_value, param_grid=param_grid_value, scoring=scroring_value, cv=cv_value)
    grid_search_model.fit(X_train, y_train)

     #Get the best model from GridSearchCV
    best_mlp_model = grid_search_model.best_estimator_

    return best_mlp_model