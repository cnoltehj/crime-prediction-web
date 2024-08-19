from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = math.sqrt(mse)
    return mse, mae, r2, rmse