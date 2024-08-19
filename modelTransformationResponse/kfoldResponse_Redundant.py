from sklearn.model_selection import train_test_split, cross_val_score, KFold

# # Define K-Fold cross-validator
# kf = KFold(n_splits=5, shuffle=True, random_state=42)

# # Example for SVR
# scores_svr = cross_val_score(svr_pipeline, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
# print("SVR Cross-Validation Scores:", scores_svr)
# print("SVR Mean Cross-Validation Score:", scores_svr.mean())

# # Example for KNN
# scores_knn = cross_val_score(knn_pipeline, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
# print("KNN Cross-Validation Scores:", scores_knn)
# print("KNN Mean Cross-Validation Score:", scores_knn.mean())

# # Example for Random Forest
# scores_rfm = cross_val_score(rfm_pipeline, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
# print("RFM Cross-Validation Scores:", scores_rfm)
# print("RFM Mean Cross-Validation Score:", scores_rfm.mean())

# # Example for MLPRegressor
# scores_mlp = cross_val_score(mlp_pipeline, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
# print("MLP Cross-Validation Scores:", scores_mlp)
# print("MLP Mean Cross-Validation Score:", scores_mlp.mean())

# # Example for XGBoost
# scores_xgb = cross_val_score(xgb_pipeline, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
# print("XGBoost Cross-Validation Scores:", scores_xgb)
# print("XGBoost Mean Cross-Validation Score:", scores_xgb.mean())
