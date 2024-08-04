from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

# Example of a pipeline for SVR with scaling
svr_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Preprocessing step
    ('model', SVR())               # Model specification
])

# Example of a pipeline for KNN with scaling
knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Preprocessing step
    ('model', KNeighborsRegressor()) # Model specification
])

# Example of a pipeline for Random Forest (scaling may not be necessary for tree-based models)
rfm_pipeline = Pipeline([
    ('model', RandomForestRegressor())  # Model specification
])

# Example of a pipeline for MLPRegressor with scaling
mlp_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Preprocessing step
    ('model', MLPRegressor())      # Model specification
])

# Example of a pipeline for XGBoost (scaling may not be necessary for tree-based models)
xgb_pipeline = Pipeline([
    ('model', XGBRegressor())  # Model specification
])
