from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import numpy as np
import pandas as pd
from Model.predictionModel import predictModel
from HelpFiles.helpers import specified_categories
from HelpFiles.helpers import data_to_plot


def prediction(model,parameter_n_estimators,parameter_max_features,parameter_min_samples_split,parameter_min_samples_leaf,parameter_random_state,parameter_criterion,parameter_bootstrap,parameter_oob_score):
   
    predictions_dict = {}   

    specified_categories = specified_categories()

    # Get the DataFrame by calling the function
    data_frame = data_to_plot()

    # Iterate over the specified categories and drop the current target category
    for category in specified_categories:
        print(f"Processing category: {category}")
        # Ensure that `data_frame` is a DataFrame
        if isinstance(data_frame, pd.DataFrame):
            # X will contain the features, excluding the current target category
            X = data_frame.drop(columns=[category])
            print(X)
            # y will contain the current target category
            y = data_frame[category]
            print(y)
        else:
            raise ValueError("data_to_plot must be a DataFrame")

        # 5. Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the Random Forest model
        model.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred = model.predict(X_test)

        # Calculate various metrics
        predictModel.set_mae = mean_absolute_error(y_test, y_pred)
        predictModel.set_mse = mean_squared_error(y_test, y_pred)
        predictModel.set_r2= r2_score(y_test, y_pred)
        predictModel.set_mape= mean_absolute_percentage_error(y_test, y_pred)
    
        # Additional metrics
        predictModel.set_tuningimprovement = ((predictModel.get_mae - predictModel.get_mse) / predictModel.get_mse) * 100  # Assuming this is the tuning improvement metric

        # Append metric values to lists
        predictModel.mae_values.append(predictModel.get_mae)
        predictModel.mse_values.append(predictModel.get_mse)
        predictModel.r2_values.append(predictModel.get_r2)
        predictModel.mape_values.append(predictModel.get_mape)
        predictModel.tuning_improvement_value.append(predictModel.get_tuningimprovement)

        # Store predicted values in the dictionary
        predictions_dict[category] = {'true_values': y_test, 'predicted_values': y_pred}
      