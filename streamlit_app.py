import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import (
    train_test_split
    )
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import altair as alt
import time
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from dataRequest.crimedbRequest import (
    fetch_all_provinces,
    fetch_policestation_per_provinces,
    fetch_stats_province_policestation,
    fetch_prediction_province_policestation,
    fetch_suggest_stats_province_policestation,
    fetch_stats_policestation_per_province
    )
from modelTransformationResponse.hyperparametersResponse import (
    hyperparameter_rfm_model,
    hyperparameter_ann_model,
    hyperparameter_knn_model,
    hyperparameter_svr_model,
    hyperparameter_xgb_model
    )
from modelTransformationResponse.paramgridsResponse import (
    param_grids_knn_model,
    param_grids_ann_model,
    param_grids_rfm_model,
    param_grids_svr_model,
    param_grids_xgb_model
)
from modelTransformationResponse.outliersResponse import (
    identify_outliers_data,
    replace_outliers_data
    )
from  modelPreprocessingResponse.modelLabelEncoder import(
    label_encode_features,
    one_hot_encode_features
)
from modelTransformationResponse.gridsearchcvResponse import mlp_gridSearchCV
from shapleyPostHocResponse.shapleyPostHocResopnse import display_shap_plots
from sklearn.metrics import (
    mean_absolute_error ,
    mean_squared_error ,
    r2_score,
    mean_absolute_percentage_error,
    adjusted_rand_score
)
from math import sqrt
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

mae_train_values, mse_train_values, r2_train_values, mape_train_values = [], [], [], []
mae_test_values, mse_test_values, r2_test_values, mape_test_values = [], [], [], []
df_crime_data_db = pd.DataFrame()
df_display_crime_data_db = pd.DataFrame() #  Currently only for shapley can be deleted if Shapley use pivot table
df_identify_outliers_db = pd.DataFrame()
df_replace_outliers_db = pd.DataFrame()
df_pivot_crime_db = pd.DataFrame()
df_provinces_db = pd.DataFrame()
df_policestations_db = pd.DataFrame()

# Dictionary to store predicted values
predictions_dict = {}
crime_categories_list = []
model_results_list = []
df_outliers_melt = ''

df_transformed_dataset = []
param_grid = []

st.set_page_config(page_title='ML Model Building', page_icon='🤖', layout='wide')

st.title('Interpretable Crime Regression ML Model Builder')

AboutTab1,DataExplorationTab2,PreditionsTab3,MetricsTab4,ShapleyAnalysisTab5,Transformationtab6,SplitScalerTab7, EncodingTab8,RealTimePreditionsTab9,RealTimeMetricsTab10 = st.tabs(['About','Data-Exploration','Preditions','Metrics','Shapley-Analysis','Transformation','Scaler-Split', 'Encoded-Data','RealTime-Preditions','RealTime-Metrics'])

with AboutTab1:
    with st.expander('About this application'):
        st.markdown('**What can this app do?**')
        st.info('This app allows users to build a machine learning (ML) model in an end-to-end workflow. Particularly, this encompasses data upload, data pre-processing, ML model building and post-model analysis.')

        st.markdown('**How to use the app?**')
        st.warning('To engage with the app, go to the sidebar and 1. Select a data set and 2. Adjust the model parameters by adjusting the various slider widgets. This will initiate the ML model building process, display the model results, and allow users to download the generated models and accompanying data.')

        st.markdown('**Under the hood**')
        st.markdown('Data sets:')
        st.code('''- SAPS statistics online dataset from their website (SAPS, 2023)
        ''', language='markdown')

        st.markdown('Libraries used:')
        st.code('''- Pandas for data wrangling
                - Scikit-learn for building a machine learning model
                - Altair for chart creation
                - Streamlit for user interface
        ''', language='markdown')

        with st.sidebar:
            st.header(f'1. Input data')

            st.markdown('**1.2. Switch from User to Developer View Modes **')
            set_development_mode = st.toggle('Switch Development Mode') # Switch to Auto Inject Development Mode

            #set_development_mode = st.toggle('Switch to Auto Inject Development Mode')

            # if set_development_mode:
                #st.markdown('**1.2. Auto Inject Parameters**')
                # set_development_not_auto_inject= st.toggle('Disable Auto Inject Mode')

            inputdatatype = st.radio('Select input data type', options=['Use database data'], index=0)

            if inputdatatype == 'Use database data':
                st.markdown('**1.1. Use database data**')
                with st.expander('Select Input Parameters'):
                    df_provinces = fetch_all_provinces()
                    province_name = st.selectbox('Select Province', df_provinces['ProvinceName'], format_func=lambda x: x, index=8)
                    province_code_value = df_provinces[df_provinces['ProvinceName'] == province_name]['ProvinceCode'].values[0]


                    # TODO add for all other provinces

                    if province_code_value == 'ZA.WC': #
                        valid_index = 110

                    df_policestations = fetch_policestation_per_provinces(province_code_value)
                    # Ensure the index is within the valid range
                    valid_index = min(0, len(df_policestations) - 1)

                    police_station_name = st.selectbox('Select Police Station', df_policestations['StationName'], format_func=lambda x: x, index=valid_index)
                    police_code_value = df_policestations[df_policestations['StationName'] == police_station_name]['StationCode'].values[0]
                    year_mapping = st.slider('Select year range from 2016 - 2023', 2023, 2016)
                    quarter_value = st.radio('Select quarter of year', options=[1, 2, 3, 4], index=0)
                    df_province_policestation_quarterly_data_db = fetch_stats_province_policestation(province_code_value, police_code_value, quarter_value)
                    df_fetch_prediction_province_policestation_data_db = fetch_prediction_province_policestation()

                    df_suggeted_province_quarterly_data_db = fetch_stats_policestation_per_province(province_code_value,quarter_value)
                    df_suggeted_province_policestation_quarterly_data_db = fetch_suggest_stats_province_policestation(province_code_value,police_code_value)

            #if set_development_mode:

                # visualise_initail_median = st.toggle('Display initial median values')
            st.markdown('**1.2. Identify outliers**')
            identify_outlier = st.toggle('Identify outliers')

            # if identify_outlier:
            df_identify_outliers_db = identify_outliers_data(df_suggeted_province_quarterly_data_db)

            st.markdown('**1.3. Replace outliers with median**')
            replace_outlier = st.toggle('Replace with Median')
            # df_replace_outliers = replace_outliers_data(df_suggeted_province_quarterly_data_db)

            st.markdown('**1.4. Set Test and Train Parameters**')
            parameter_split_size = st.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

            st.subheader('2. Select Algorithm')
            with st.expander('Algorithms'):
                algorithm = st.radio('', options=['ANN (MLPRegressor)', 'KNN', 'RFM', 'SVR','XGBoost'], index=2)

            st.subheader('3. Learning Parameters')
            with st.expander('See parameters', expanded=False):

                if algorithm in ['RFM' , 'XGBoost']:
                    parameter_n_estimators = st.slider('Number of estimators (n_estimators)', 10, 50, 100)  #1000
                
                if algorithm in ['RFM' , 'XGBoost']:
                    # st.subheader('4. General Parameters')
                    # with st.expander('See parameters', expanded=False):
                    parameter_random_state = st.slider('Seed number (random_state)', 0, 1000, 42, 1)

                if algorithm == 'RFM':
                        parameter_max_features = st.select_slider('Max features (max_features)', options=['sqrt', 'log2'])
                        parameter_min_samples_split = st.slider('Minimum number of samples required to split an internal node (min_samples_split)', 1, 2, 5, 10)
                        parameter_min_samples_leaf = st.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)
                        parameter_criterion = st.select_slider('Performance measure (criterion)', options=['squared_error', 'absolute_error', 'friedman_mse', 'poisson'])
                        parameter_bootstrap = st.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
                        parameter_oob_score = st.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])
                elif algorithm == 'ANN (MLPRegressor)':
                        parameter_hidden_layer_size = st.select_slider('Hidden layers size is the  number of neorons in each hidden layer (hidden_layer_size)', options=[(50, 50), (100,)])
                        parameter_solver = st.select_slider('Solver for weight optimization (solver) ', options=['adam', 'sgd'])
                        parameter_activation = st.select_slider('Activation function for the hidden layer (activation)', options=['tanh', 'relu'])
                elif algorithm == 'KNN':
                        parameter_n_neighbors = st.select_slider('Number of neighbors to use (n_neighbors )', options= [3, 5, 7])
                        parameter_weights = st.select_slider('Weight function used in prediction (weights)', options=['uniform', 'distance'])
                elif algorithm == 'SVR':
                        parameter_kernel = st.select_slider('Specifies the kernel type to be (kernel)', options=['linear', 'rbf'])
                        parameter_C = st.select_slider('Regularization parameter (C)', options=[1,10])
                        parameter_epsilon = st.select_slider('Epsilon in the epsilon-SVR (epsilon)', options=[0.1 , 0.2])
                elif algorithm == 'XGBoost':
                        parameter_learning_rate = st.select_slider('Boosting learning rate (learning_rate)', options=[0.01, 0.1])
                        parameter_max_depth = st.select_slider('Maximum depth of a tree (max_depth)', options=[3,5,7])
                        parameter_min_child_weight = st.select_slider('Minimum sum of instance weight (hessian) needed in a child (min_child_weight)', options=[1,3])
                        parameter_subsample = st.select_slider('Subsample ratio of the training instances (subsample)', options=[0.8, 1.0])
                        parameter_cosample_bytree = st.select_slider('Subsample ratio of columns when constructing each tree (cosample_bytree)', options=[0.8, 1.0])

                sleep_time = st.slider('Sleep time', 0, 3, 0)

        if not df_crime_data_db.empty:
            with st.status("Running ...", expanded=True) as status:

                # st.write("Loading data ...")
                # time.sleep(sleep_time)

                # st.write("Preparing data ...")
                # time.sleep(sleep_time)

                # Initialize empty lists for metrics and crime categories
                crime_categories_list = df_crime_data_db['CrimeCategory'].tolist()

    if not set_development_mode:
        with st.expander(f'Prediction Values', expanded=False):
            visualise_prediction = st.toggle('Prediction Values dataset')
            st.dataframe(df_fetch_prediction_province_policestation_data_db, height=280, use_container_width=True)
            df_predtions_melt = df_fetch_prediction_province_policestation_data_db.melt(id_vars=['CrimeCategory'], var_name='Year', value_name='Percentage')

            if visualise_prediction:
                # Create the graph using seaborn
                plt.figure(figsize=(10, 6))
                sns.lineplot(data=df_predtions_melt, x='Year', y='Percentage', hue='CrimeCategory', marker='o')
                plt.title('Crime Trends Over the Years')
                plt.legend(title='Crime Category', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.xticks(rotation=45)

                        # Annotate each data point with its value
                for i in range(df_predtions_melt.shape[0]):
                    plt.text(df_predtions_melt['Year'].iloc[i], df_predtions_melt['Percentage'].iloc[i],
                    f"{df_predtions_melt['Percentage'].iloc[i]:.2f}", color='black', ha='right', va='bottom')

                # Display the graph in Streamlit
                st.pyplot(plt)
with DataExplorationTab2:
    with st.expander(f'Initial dataset', expanded=False):
        visualise_initialdate = st.toggle('Visualise initial dataset')
        st.dataframe(df_suggeted_province_quarterly_data_db.sort_values(by='PoliceStationCode'), height=280, use_container_width=True)

        # Melting the DataFrame to get a tidy format
        df_initial_data_melt = df_suggeted_province_quarterly_data_db.melt(
            id_vars=['CrimeCategory', 'ProvinceCode', 'PoliceStationCode','Quarter'],
            var_name='Year',
            value_name='Percentage'
        )

        # Setting crime data DataFrame
        #df_crime_data_db = df_suggeted_province_quarterly_data_db.sort_values(by='df_identify_outliers_db_sort')

        if visualise_initialdate:
            # Create the graph using seaborn
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=df_initial_data_melt, x='Year', y='Percentage', hue='CrimeCategory', marker='o')
            plt.title('Crime Trends Over the Years')
            plt.xlabel('Year')
            plt.ylabel('Percentage')
            plt.legend(title='Crime Category', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45)

            # Annotate each data point with its value
            for i in range(df_initial_data_melt.shape[0]):
                plt.text(
                    df_initial_data_melt['Year'].iloc[i],
                    df_initial_data_melt['Percentage'].iloc[i],
                    f"{df_initial_data_melt['Percentage'].iloc[i]:.2f}",
                    color='black',
                    ha='right',
                    va='bottom'
                )

            # Display the graph in Streamlit
            st.pyplot(plt)

    #if set_development_mode:
with PreditionsTab3:
    with st.expander('Best Predictions', expanded=False):
        performance_col = st.columns((2, 0.2, 3))

with MetricsTab4:
    with st.expander('Best Predictions', expanded=False):
        performance_col = st.columns((2, 0.2, 3))
       
with ShapleyAnalysisTab5:
    with st.expander('Shapley Post-Hoc Analysis', expanded=False):
        performance_col = st.columns((2, 0.2, 3))

with Transformationtab6:
    # if identify_outlier:
    with st.expander('Identify outliers', expanded=False):
        performance_col = st.columns((2, 0.2, 3))

        with performance_col[0]:
            st.header('Outliers', divider='rainbow')
            st.dataframe(df_identify_outliers_db.sort_values(by='PoliceStationCode'))

                # Plot box plot of the data with outliers replaced
        with performance_col[2]:
            st.header('Outliers percentage plot', divider='rainbow')
            # Melt DataFrame to long format for easy plotting
            df_identify_outliers_melted = df_identify_outliers_db.melt(id_vars=['CrimeCategory', 'ProvinceCode', 'PoliceStationCode', 'Quarter', 'Outliers'],
            var_name='Year', value_name='Percentage')

            # Convert 'Outliers' to list of outlier values
            def parse_outliers(outliers):
                if isinstance(outliers, str):
                    return [float(i) for i in outliers.split(',')]
                elif isinstance(outliers, (float, int)):
                    return [float(outliers)]
                return []

            df_identify_outliers_melted['Outliers'] = df_identify_outliers_melted['Outliers'].apply(parse_outliers)

            df_exploded = df_identify_outliers_melted.explode('Outliers')
            
            plt.figure(figsize=(12, 8))
            # Plotting the boxplot
            sns.boxplot(x='Year', y='Percentage', data=df_identify_outliers_melted)
            plt.title("Box Plot Identifying the Outliers")
            plt.xticks(rotation=45)

            # Add annotations for outliers
            for _, row in df_exploded.iterrows():
                plt.text(
                    x=row['Year'],
                    y=row['Outliers'] + 2,  # Adjust this to fit your plot
                    s=f"{row['Outliers']:.1f}",
                    fontsize=9,
                    color='black',
                    ha='center'
                )

            st.pyplot(plt)

        # if replace_outlier:
        #     if not (df_replace_outliers_db.empty and df_identify_outliers_db.empty) and replace_outlier:
    with st.expander('Replaced outliers with the median value', expanded=False):
        visualise_replace_outliers = st.toggle('Visualise outliers dataset replaced by median value')

        df_replace_outliers = replace_outliers_data(df_suggeted_province_quarterly_data_db)
        st.dataframe(df_replace_outliers.sort_values(by='PoliceStationCode'), height=210, use_container_width=True)

        if not df_replace_outliers.empty:
            # Melt DataFrame for visualization
            df_outliers_replaced_melt = df_replace_outliers.melt(
            id_vars=['CrimeCategory', 'ProvinceCode', 'PoliceStationCode', 'Quarter'],
            var_name='Year',
            value_name='Percentage'
            )
            
            if visualise_replace_outliers:
                            # Create the graph using seaborn
                plt.figure(figsize=(12, 8))
                sns.lineplot(data=df_outliers_replaced_melt, x='Year', y='Percentage', hue='CrimeCategory', marker='o')
                plt.title('Crime Trends Over the Years with Outliers Replaced')
                plt.xlabel('Year')
                plt.ylabel('Percentage')
                plt.legend(title='Crime Category', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.xticks(rotation=45)

                                # Annotate each data point with its value
                for i in range(df_outliers_replaced_melt.shape[0]):
                    plt.text(
                        x=df_outliers_replaced_melt['Year'].iloc[i],
                        y=df_outliers_replaced_melt['Percentage'].iloc[i],
                        s=f"{df_outliers_replaced_melt['Percentage'].iloc[i]:.2f}",
                        color='black',
                        ha='right',
                        va='bottom'
                    )

                    # Adjust layout to make room for the legend
                    plt.tight_layout()

                    # Display the graph
                st.pyplot(plt)

    # C - Splitting the Data =========

            # Initialize the models
    models = {
        'RFM': RandomForestRegressor(),
        'SVR': SVR(),
        'XGBR': XGBRegressor(),
        'KNNR': KNeighborsRegressor(),
        'MLPR': MLPRegressor()
    }

    # Define the parameters grid for each model (these are examples, you can customize them)
    params = {
    'RFM': {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]},
    'SVR': {'C': [1, 10], 'kernel': ['linear', 'rbf']},
    'XGBR': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]},
    'KNNR': {'n_neighbors': [5, 10], 'weights': ['uniform', 'distance']},
    'MLPR': {'hidden_layer_sizes': [(100,), (100, 50)], 'activation': ['relu', 'tanh']}
}

    def evaluate_metrics(y_true, y_pred):
        metrics_list = {
            'MAE': mean_absolute_error(y_true, y_pred),
            'MSE': mean_squared_error(y_true, y_pred),
            'R²': r2_score(y_true, y_pred),
            'MAPE': mean_absolute_percentage_error(y_true, y_pred)
        }
        return metrics_list

    # Perform GridSearchCV for each model
    best_estimators_train = {}
    best_estimators_test = {}
    metrics = {}
    scoring_value = 'neg_mean_squared_error'
    cv_value = 3

#==========================
# A - Keeping dataset in wide format for prediction =====
# Assuming df_replace_outliers is the original dataset with years as columns
df_wide = df_replace_outliers.copy()

# B - Defining feature set X (years 2016-2023) and target y (2024 prediction) =====
# We exclude 'CrimeCategory', 'ProvinceCode', 'PoliceStationCode', 'Quarter' as these are categorical
X = df_wide[['2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']]
y = df_wide['2023']  # Using 2023 as a proxy for training, predicting 2024

# C - Splitting the dataset into train/test sets (use 2016-2023 data for training) =====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensure X_train and X_test are DataFrames before scaling
X_train_display = pd.DataFrame(X_train, columns=X.columns)
X_test_display = pd.DataFrame(X_test, columns=X.columns)

# D - Scaling the numeric features ===== Use values for Predicting withou Encoding
scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

predictions = {}

# Loop through each model and perform GridSearchCV
for name, model in models.items():
    print(f"Training {name} model...")
    
    # Get the parameter grid for the current model
    param_grid = params[name]
    
    # Define GridSearchCV
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    
    # Fit the model using GridSearchCV
    grid_search.fit(X_train_scaled, y_train)
    
    # Retrieve the best estimator (model with best parameters)
    best_model = grid_search.best_estimator_
    
    # Predict on the test set
    y_pred = best_model.predict(X_test_scaled)
    
    # Store predictions and calculate metrics
    predictions[name] = {
        'Prediction': y_pred,
        'True_value': y_test,
        'MSE': mean_squared_error(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'R²': r2_score(y_test, y_pred),
        'MAPE': mean_absolute_percentage_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
    }

# F - Merging predictions with original dataset =====
# Create a DataFrame for the output format
output_data = df_wide[['CrimeCategory', 'ProvinceCode', 'PoliceStationCode', 'Quarter', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']].copy()

output_mertics = df_wide[['CrimeCategory', 'ProvinceCode', 'PoliceStationCode', 'Quarter', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']].copy()

# Adding predictions and true values for each model
for name, results in predictions.items():
    # Adding prediction for 2024 as a new column
    output_data[f'Prediction_{name}'] = pd.Series(results['Prediction'], index=y_test.index)
    
    # Adding true values for 2023 (actual values from the test set)
    output_data[f'True_value_{name}'] = pd.Series(y_test.values, index=y_test.index)

    output_mertics[f'MSE_{name}'] = pd.Series(results['MSE'], index=y_test.index)
    



# G - Final output with predictions =====
# st.write("Final Output with Predictions:")
# st.dataframe(output_data)


#++++++++++++++++++++++++++
#+++++++++++++++++++++++++
# Scenario 1: Label encoding for 'PoliceStationCode' and 'Quarter' - Training and Prediction

# Initialize Label Encoders for categorical variables
label_encoder_psc = LabelEncoder()
label_encoder_qtr = LabelEncoder()

# Copy data for encoding
X_train_encoded = X_train_display.copy()
X_test_encoded = X_test_display.copy()

# Check if 'PoliceStationCode' and 'Quarter' exist in both train and test sets before encoding
if 'PoliceStationCode' in X_train_encoded.columns and 'PoliceStationCode' in X_test_encoded.columns:
    X_train_encoded['PoliceStationCode'] = label_encoder_psc.fit_transform(X_train_encoded['PoliceStationCode'])
    X_test_encoded['PoliceStationCode'] = label_encoder_psc.transform(X_test_encoded['PoliceStationCode'])
else:
    print("Warning: 'PoliceStationCode' column is missing from the dataset.")

if 'Quarter' in X_train_encoded.columns and 'Quarter' in X_test_encoded.columns:
    X_train_encoded['Quarter'] = label_encoder_qtr.fit_transform(X_train_encoded['Quarter'])
    X_test_encoded['Quarter'] = label_encoder_qtr.transform(X_test_encoded['Quarter'])
else:
    print("Warning: 'Quarter' column is missing from the dataset.")

# Select only numerical columns for scaling
numerical_columns = X_train_encoded.select_dtypes(include=[np.number]).columns
X_train_numerical = X_train_encoded[numerical_columns]
X_test_numerical = X_test_encoded[numerical_columns]

# Apply StandardScaler to numerical features
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train_numerical)
X_test_scaled = scaler_X.transform(X_test_numerical)

# Scale the target variable y
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

# Initialize dictionaries for storing metrics and predictions
metrics = {}
predictions = {}

# Iterate over each unique CrimeCategory in the training dataset
for category in df_replace_outliers['CrimeCategory'].unique():
    print(f"Processing CrimeCategory: {category}")
    
    # Filter the training data for the current category
    category_mask = df_replace_outliers['CrimeCategory'] == category
    X_category_train = X_train_scaled[category_mask]
    y_category_train = y_train_scaled[category_mask]

    # # Ensure data consistency: Skip the category if lengths of X and y don't match
    # if len(X_category_train) != len(y_category_train):
    #     print(f"Skipping category {category} due to inconsistent sample sizes.")
    #     continue
    
    # # Check if there's enough data for training
    # if len(X_category_train) < 2 or len(y_category_train) < 2:
    #     print(f"Skipping category {category} due to insufficient data.")
    #     continue
    
    # Initialize dictionaries to store category-specific results
    predictions_category = {}
    metrics_category = {}

    # Iterate over each model and perform GridSearchCV
    for model_name, model in models.items():
        print(f"Training {model_name} for {category}...")

        # Perform GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(estimator=model, param_grid=params[model_name], cv=3, n_jobs=-1, scoring='r2')
        grid_search.fit(X_category_train, y_category_train.ravel())  # Flatten the target for training

        # Store the best model found by GridSearchCV
        best_model = grid_search.best_estimator_

        # Predict on the test set (use the filtered test data)
        y_pred_scaled = best_model.predict(X_test_scaled)

        # Inverse transform the predictions to the original scale
        y_pred_original = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_test_original = scaler_y.inverse_transform(y_test_scaled).flatten()

        # Store the predictions for the current category and model
        predictions[model_name] = y_pred_original

        # Calculate evaluation metrics
        mse = mean_squared_error(y_test_original, y_pred_original)
        mae = mean_absolute_error(y_test_original, y_pred_original)
        r2 = r2_score(y_test_original, y_pred_original)
        rmse = np.sqrt(mse)
        
        # Store the metrics for the current model and category
        metrics[model_name] = {
            'Algorithm': model_name,
            'CrimeCategory': category,
            'MSE': mse,
            'MAE': mae,
            'R²': r2,
            'RMSE': rmse
        }

        # Display the calculated metrics
        print(f"Metrics for {model_name} (Category: {category}):")
        print(f"  MSE: {mse}")
        print(f"  MAE: {mae}")
        print(f"  R²: {r2}")
        print(f"  RMSE: {rmse}")
        print("\n" + "-"*50 + "\n")

    # Store predictions and metrics per category
    predictions[category] = predictions_category
    metrics[category] = metrics_category

# Convert metrics to a DataFrame for analysis
metrics_df = pd.DataFrame(metrics)

# Display the results DataFrame
print(metrics_df)

# #+++++++++++++++++++++++++++++++++++++++++++++
# #++++++++++++++++++++++++++++++++++++++++++++++

# # Scenario 2 - One-hot encoding for 'PoliceStationCode' and 'Quarter'
# onehot_encoder_train_2 = OneHotEncoder()

# # Copy the training data
# data_onehot_encoded_train_2 = X_train_display.copy()

# # Apply One-hot encoding to 'PoliceStationCode' and 'Quarter' in the training set
# encoded_features_onehot_encoded_train_2 = onehot_encoder_train_2.fit_transform(
#     data_onehot_encoded_train_2[['PoliceStationCode', 'Quarter']]).toarray()

# encoded_df_onehot_encoded_train_2 = pd.DataFrame(
#     encoded_features_onehot_encoded_train_2,
#     columns=onehot_encoder_train_2.get_feature_names_out(['PoliceStationCode', 'Quarter'])
# )

# data_onehot_encoded_train_2 = pd.concat(
#     [data_onehot_encoded_train_2, encoded_df_onehot_encoded_train_2], axis=1
# ).drop(['PoliceStationCode', 'Quarter'], axis=1)

# # Apply the same encoding to the test set
# data_onehot_encoded_test_2 = X_test_display.copy()

# encoded_features_onehot_encoded_test_2 = onehot_encoder_train_2.transform(
#     data_onehot_encoded_test_2[['PoliceStationCode', 'Quarter']]).toarray()

# encoded_df_onehot_encoded_test_2 = pd.DataFrame(
#     encoded_features_onehot_encoded_test_2,
#     columns=onehot_encoder_train_2.get_feature_names_out(['PoliceStationCode', 'Quarter'])
# )

# data_onehot_encoded_test_2 = pd.concat(
#     [data_onehot_encoded_test_2, encoded_df_onehot_encoded_test_2], axis=1
# ).drop(['PoliceStationCode', 'Quarter'], axis=1)

# # Select only numerical columns
# numerical_columns_onehot_encoded_2 = data_onehot_encoded_train_2.select_dtypes(include=[np.number]).columns
# data_numerical_onehot_only_train_2 = data_onehot_encoded_train_2[numerical_columns_onehot_encoded_2]
# data_numerical_onehot_only_test_2 = data_onehot_encoded_test_2[numerical_columns_onehot_encoded_2]

# # Apply StandardScaler
# scaler_X_2 = StandardScaler()
# X_onehot_train_scaled_2 = scaler_X_2.fit_transform(data_numerical_onehot_only_train_2)
# X_onehot_test_scaled_2 = scaler_X_2.transform(data_numerical_onehot_only_test_2)

# # Use a separate scaler for y (target variable)
# scaler_y_2 = StandardScaler()
# y_train_scaled_2 = scaler_y_2.fit_transform(y_train_reshaped)  # Reshaped target variable for scaling
# y_test_scaled_2 = scaler_y_2.transform(y_test_reshaped)

# # Train and Evaluate
# metrics_scenario_2 = {}
# best_models_scenario_2 = {}
# predictions_scenario_2 = {}

# # Before fitting the model, ensure that X_onehot_train_scaled and y_train_scaled are of the same length
# assert len(X_onehot_train_scaled_2) == len(y_train_scaled_2), "Mismatch in the length of X_onehot_train_scaled and y_train_scaled"

# # Initialize dictionaries to store predictions and metrics for each category
# predictions_per_category_scenario_2 = {}
# metrics_per_category_scenario_2 = {}

# # Iterate over each unique CrimeCategory in the dataset
# for category in df_replace_outliers['CrimeCategory'].unique():
#     print(f"Processing CrimeCategory: {category}")
    
#     # Filter the dataset for the current category
#     X_category = X_train_scaled_df[df_replace_outliers['CrimeCategory'] == category]
    
#     # Drop the specified columns if they exist in the DataFrame
#     columns_to_drop_2 = ['CrimeCategory', 'ProvinceCode', 'PoliceStationCode', 'Quarter']
#     X_category_2 = X_category.drop(columns=[col for col in columns_to_drop_2 if col in X_category.columns])
    
#     # Filter the target variable
#     y_category_2 = y_train[df_replace_outliers['CrimeCategory'] == category]

#     # Check if the lengths of X_category and y_category are consistent
#     if len(X_category_2) != len(y_category_2):
#         print(f"Skipping category {category} due to inconsistent sample sizes.")
#         continue
    
#     # Check if there's enough data to train the model
#     if len(X_category_2) < 2 or len(y_category_2) < 2:
#         print(f"Skipping category {category} due to insufficient data.")
#         continue
    
#     # Initialize dictionaries to store results for the current category
#     predictions_scenario_2_category = {}
#     metrics_scenario_2_category = {}
    
#     # Iterate over each model, perform GridSearchCV, and make predictions
#     for name, model in models.items():
#         print(f"Training and predicting with {name} for {category}...")

#         # For KNeighborsRegressor, adjust n_neighbors based on the available samples
#         if name == 'KNNR':
#             n_samples_fit = len(X_category_2)
#             if n_samples_fit < 5:  # If fewer than 5 samples, adjust n_neighbors to n_samples_fit
#                 params['KNNR']['n_neighbors'] = [n_samples_fit]
        
#         # Perform GridSearchCV to find the best parameters
#         grid_search_2 = GridSearchCV(estimator=model, param_grid=params[name], cv=2, n_jobs=-1, scoring='r2')
#         grid_search_2.fit(X_category, y_category_2)
        
#         # Store the best model
#         best_models_scenario_2[name] = grid_search_2.best_estimator_

#         # Prepare X_test_scaled for prediction by excluding the categorical columns
#         X_test_scaled_filtered_2 = X_test_scaled_df.drop(columns=[col for col in columns_to_drop_2 if col in X_test_scaled_df.columns])
        
#         # Make predictions on the test set
#         y_pred_2_scaled = grid_search_2.predict(X_test_scaled_filtered_2)
       
#          # Convert predictions back to original scale using y_scaler (the scaler applied to the target variable)
#          # Inverse transform predictions back to original scale
#         y_pred_original_2 = scaler_y_2.inverse_transform(y_pred_2_scaled.reshape(-1, 1)).flatten()
#         y_test_original_2 = scaler_y_2.inverse_transform(y_test_scaled_2.reshape(-1, 1)).flatten()

#         # Store the predictions in the original values for the current category
#         predictions_scenario_2[name] = y_pred_original_2

#         # Calculate evaluation metrics
#         mse_scenario_2 = mean_squared_error(y_test_original_2, y_pred_original_2)
#         mae_scenario_2 = mean_absolute_error(y_test_original_2, y_pred_original_2)
#         r2_scenario_2 = r2_score(y_test_original_2, y_pred_original_2)
#         rmse_scenario_2 = np.sqrt(mse_scenario_2)
        
#         # Store the metrics
#         metrics_scenario_2[name] = {
#             'CrimeCategory': X_test_categories['CrimeCategory'],
#             'ProvinceCode': X_test_categories['ProvinceCode'],
#             'PoliceStationCode': X_test_categories['PoliceStationCode'],
#             'Quarter': X_test_categories['Quarter'],
#             'MSE': mse_scenario_2,
#             'MAE': mae_scenario_2,
#             'R²': r2_scenario_2,
#             'RMSE': rmse_scenario_2
#         }

#         metrics_scenario_2_df = pd.DataFrame(metrics_scenario_2)
        
#         # Display the metrics
#         print(f"Metrics scenario_2 for {name}:")
#         print(f"  MSE: {mse_scenario_2}")
#         print(f"  MAE: {mae_scenario_2}")
#         print(f"  R²: {r2_scenario_2}")
#         print(f"  RMSE: {rmse_scenario_2}")
#         print("\n" + "-"*50 + "\n")


#     predicted_values_2 = predictions_scenario_2[name] # predictions_scenario_2_category[model_name]  # Adjust 'model_name' accordingly 

#     # Ensure 'predicted_values' is a list or a Series before creating the DataFrame
#     predicted_values_2 = pd.Series(predicted_values_2)

#     # Now create the DataFrame with all columns properly aligned
#     results_2_df = pd.DataFrame({
#         'CrimeCategory': X_test_categories['CrimeCategory'],
#         'ProvinceCode': X_test_categories['ProvinceCode'],
#         'PoliceStationCode': X_test_categories['PoliceStationCode'],
#         'Quarter': X_test_categories['Quarter'],
#         'TrueValue': y_test_scaled_2.flatten(),
#         'PredictedValue': predicted_values_2
#     })

#     # Store predictions and metrics for the current category
#     predictions_per_category_scenario_2[category] = predictions_scenario_2_category
#     metrics_per_category_scenario_2[category] = metrics_scenario_2_category

# #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#    # Scenario 3 - Label encoding for 'PoliceStationCode' and One-hot encoding for 'Quarter'

# # Label encode 'PoliceStationCode'
# label_encoder_police_station_3 = LabelEncoder()
# data_label_onehot_train_3 = X_train_display.copy()

# # Apply Label encoding for 'PoliceStationCode' in the training set
# data_label_onehot_train_3['PoliceStationCode'] = label_encoder_police_station_3.fit_transform(data_label_onehot_train_3['PoliceStationCode'])

# # One-hot encode 'Quarter'
# onehot_encoder_quarter_3 = OneHotEncoder()

# # Apply One-hot encoding for 'Quarter' in the training set
# encoded_features_quarter_train_3 = onehot_encoder_quarter_3.fit_transform(data_label_onehot_train_3[['Quarter']]).toarray()

# encoded_df_quarter_train_3 = pd.DataFrame(
#     encoded_features_quarter_train_3,
#     columns=onehot_encoder_quarter_3.get_feature_names_out(['Quarter'])
# )

# data_label_onehot_train_3 = pd.concat(
#     [data_label_onehot_train_3, encoded_df_quarter_train_3], axis=1
# ).drop(['Quarter'], axis=1)

# # Apply the same encoding to the test set
# data_label_onehot_test_3 = X_test_display.copy()

# # Label encode 'PoliceStationCode' in the test set
# data_label_onehot_test_3['PoliceStationCode'] = label_encoder_police_station_3.transform(data_label_onehot_test_3['PoliceStationCode'])

# # One-hot encode 'Quarter' in the test set
# encoded_features_quarter_test_3 = onehot_encoder_quarter_3.transform(data_label_onehot_test_3[['Quarter']]).toarray()

# encoded_df_quarter_test_3 = pd.DataFrame(
#     encoded_features_quarter_test_3,
#     columns=onehot_encoder_quarter_3.get_feature_names_out(['Quarter'])
# )

# data_label_onehot_test_3 = pd.concat(
#     [data_label_onehot_test_3, encoded_df_quarter_test_3], axis=1
# ).drop(['Quarter'], axis=1)

# # Select only numerical columns
# numerical_columns_label_onehot_3 = data_label_onehot_train_3.select_dtypes(include=[np.number]).columns
# data_numerical_label_onehot_train_3 = data_label_onehot_train_3[numerical_columns_label_onehot_3]
# data_numerical_label_onehot_test_3 = data_label_onehot_test_3[numerical_columns_label_onehot_3]

# # Apply StandardScaler
# scaler_X_3 = StandardScaler()
# X_label_onehot_train_scaled_3 = scaler_X_3.fit_transform(data_numerical_label_onehot_train_3)
# X_label_onehot_test_scaled_3 = scaler_X_3.transform(data_numerical_label_onehot_test_3)

# # Use a separate scaler for y (target variable)
# scaler_y_3 = StandardScaler()
# y_train_scaled_3 = scaler_y_3.fit_transform(y_train_reshaped)  # Reshaped target variable for scaling
# y_test_scaled_3 = scaler_y_3.transform(y_test_reshaped)

# # Train and Evaluate
# metrics_scenario_3 = {}
# best_models_scenario_3 = {}
# predictions_scenario_3 = {}

# # Before fitting the model, ensure that X_label_onehot_train_scaled and y_train_scaled are of the same length
# assert len(X_label_onehot_train_scaled_3) == len(y_train_scaled_3), "Mismatch in the length of X_label_onehot_train_scaled and y_train_scaled"

# # Initialize dictionaries to store predictions and metrics for each category
# predictions_per_category_scenario_3 = {}
# metrics_per_category_scenario_3 = {}

# # Iterate over each unique CrimeCategory in the dataset
# for category in df_replace_outliers['CrimeCategory'].unique():
#     print(f"Processing CrimeCategory: {category}")
    
#     # Filter the dataset for the current category
#     X_category_3 = X_train_scaled_df[df_replace_outliers['CrimeCategory'] == category]
    
#     # Drop the specified columns if they exist in the DataFrame
#     columns_to_drop_3 = ['CrimeCategory', 'ProvinceCode', 'PoliceStationCode', 'Quarter']
#     X_category_3 = X_category_3.drop(columns=[col for col in columns_to_drop_3 if col in X_category_3.columns])
    
#     # Filter the target variable
#     y_category_3 = y_train[df_replace_outliers['CrimeCategory'] == category]

#     # Check if the lengths of X_category and y_category are consistent
#     if len(X_category) != len(y_category_3):
#         print(f"Skipping category {category} due to inconsistent sample sizes.")
#         continue
    
#     # Check if there's enough data to train the model
#     if len(X_category_3) < 2 or len(y_category_3) < 2:
#         print(f"Skipping category {category} due to insufficient data.")
#         continue
    
#     # Initialize dictionaries to store results for the current category
#     predictions_scenario_3_category = {}
#     metrics_scenario_3_category = {}
    
#     # Iterate over each model, perform GridSearchCV, and make predictions
#     for name, model in models.items():
#         print(f"Training and predicting with {name} for {category}...")

#         # For KNeighborsRegressor, adjust n_neighbors based on the available samples
#         if name == 'KNNR':
#             n_samples_fit = len(X_category_3)
#             if n_samples_fit < 5:  # If fewer than 5 samples, adjust n_neighbors to n_samples_fit
#                 params['KNNR']['n_neighbors'] = [n_samples_fit]
        
#         # Perform GridSearchCV to find the best parameters
#         grid_search_3 = GridSearchCV(estimator=model, param_grid=params[name], cv=2, n_jobs=-1, scoring='r2')
#         grid_search_3.fit(X_category, y_category_3)
        
#         # Store the best model
#         best_models_scenario_3[name] = grid_search_3.best_estimator_

#         # Prepare X_test_scaled for prediction by excluding the categorical columns
#         X_test_scaled_filtered_3 = X_test_scaled_df.drop(columns=[col for col in columns_to_drop_3 if col in X_test_scaled_df.columns])
        
#         # Make predictions on the test set
#         y_pred_scaled_3 = grid_search_3.predict(X_test_scaled_filtered_3)
        
#         # Inverse transform predictions back to original scale
#         y_pred_original_3 = scaler_y_3.inverse_transform(y_pred_scaled_3.reshape(-1, 1)).flatten()
#         y_test_original_3 = scaler_y_3.inverse_transform(y_test_scaled_3.reshape(-1, 1)).flatten()
#         predictions_scenario_3[name] = y_pred_original_3

#         # Calculate evaluation metrics
#         mse_scenario_3 = mean_squared_error( y_test_original_3, y_pred_original_3)
#         mae_scenario_3 = mean_absolute_error( y_test_original_3, y_pred_original_3)
#         r2_scenario_3 = r2_score(y_test_original_3, y_pred_original_3)
#         rmse_scenario_3 = np.sqrt(mse_scenario_3)
        
#         # Store the metrics
#         metrics_scenario_3[name] = {
#             'CrimeCategory': X_test_categories['CrimeCategory'],
#             'ProvinceCode': X_test_categories['ProvinceCode'],
#             'PoliceStationCode': X_test_categories['PoliceStationCode'],
#             'Quarter': X_test_categories['Quarter'],
#             'MSE': mse_scenario_3,
#             'MAE': mae_scenario_3,
#             'R²': r2_scenario_3,
#             'RMSE': rmse_scenario_3
#         }

#         metrics_scenario_3_df = pd.DataFrame(metrics_scenario_3)
        
#         # Display the metrics
#         print(f"Metrics scenario_3 for {name}:")
#         print(f"  MSE: {mse_scenario_3}")
#         print(f"  MAE: {mae_scenario_3}")
#         print(f"  R²: {r2_scenario_3}")
#         print(f"  RMSE: {rmse_scenario_3}")
#         print("\n" + "-"*50 + "\n")

#     predicted_values_3 = predictions_scenario_3[name] 

#     # Ensure 'predicted_values' is a list or a Series before creating the DataFrame
#     predicted_values_3 = pd.Series(predicted_values_3)

#     # Now create the DataFrame with all columns properly aligned
#     results_3_df = pd.DataFrame({
#         'CrimeCategory': X_test_categories['CrimeCategory'],
#         'ProvinceCode': X_test_categories['ProvinceCode'],
#         'PoliceStationCode': X_test_categories['PoliceStationCode'],
#         'Quarter': X_test_categories['Quarter'],
#         'TrueValue': y_test_scaled_3.flatten(),
#         'PredictedValue': predicted_values_3
#     })

#     # Store predictions and metrics for the current category
#     predictions_per_category_scenario_3[category] = predictions_scenario_3_category
#     metrics_per_category_scenario_3[category] = metrics_scenario_3_category

# #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#    # Scenario 4 - Label encoding for 'Quarter' and One-hot encoding for 'PoliceStationCode'

# # Label encode 'Quarter'
# label_encoder_quarter_4 = LabelEncoder()
# data_label_onehot_train_4 = X_train_display.copy()

# # Apply Label encoding for 'Quarter' in the training set
# data_label_onehot_train_4['Quarter'] = label_encoder_quarter_4.fit_transform(data_label_onehot_train_4['Quarter'])

# # One-hot encode 'PoliceStationCode'
# onehot_encoder_police_station_4 = OneHotEncoder()

# # Apply One-hot encoding for 'PoliceStationCode' in the training set
# encoded_features_police_train_4 = onehot_encoder_police_station_4.fit_transform(data_label_onehot_train_4[['PoliceStationCode']]).toarray()

# encoded_df_police_train_4 = pd.DataFrame(
#     encoded_features_police_train_4,
#     columns=onehot_encoder_police_station_4.get_feature_names_out(['PoliceStationCode'])
# )

# data_label_onehot_train_4 = pd.concat(
#     [data_label_onehot_train_4, encoded_df_police_train_4], axis=1
# ).drop(['PoliceStationCode'], axis=1)

# # Apply the same encoding to the test set
# data_label_onehot_test_4 = X_test_display.copy()

# # Label encode 'Quarter' in the test set
# data_label_onehot_test_4['Quarter'] = label_encoder_quarter_4.transform(data_label_onehot_test_4['Quarter'])

# # One-hot encode 'PoliceStationCode' in the test set
# encoded_features_police_test_4 = onehot_encoder_police_station_4.transform(data_label_onehot_test_4[['PoliceStationCode']]).toarray()

# encoded_df_police_test_4 = pd.DataFrame(
#     encoded_features_police_test_4,
#     columns=onehot_encoder_police_station_4.get_feature_names_out(['PoliceStationCode'])
# )

# data_label_onehot_test_4 = pd.concat(
#     [data_label_onehot_test_4, encoded_df_police_test_4], axis=1
# ).drop(['PoliceStationCode'], axis=1)

# # Select only numerical columns
# numerical_columns_label_onehot_4 = data_label_onehot_train_4.select_dtypes(include=[np.number]).columns
# data_numerical_label_onehot_train_4 = data_label_onehot_train_4[numerical_columns_label_onehot_4]
# data_numerical_label_onehot_test_4 = data_label_onehot_test_4[numerical_columns_label_onehot_4]

# # Apply StandardScaler
# scaler_X_4 = StandardScaler()
# X_label_onehot_train_scaled_4 = scaler_X_4.fit_transform(data_numerical_label_onehot_train_4)
# X_label_onehot_test_scaled_4 = scaler_X_4.transform(data_numerical_label_onehot_test_4)

# # Use a separate scaler for y (target variable)
# scaler_y_4 = StandardScaler()
# y_train_scaled_4 = scaler_y_4.fit_transform(y_train_reshaped)  # Reshaped target variable for scaling
# y_test_scaled_4 = scaler_y_4.transform(y_test_reshaped)

# # Train and Evaluate
# metrics_scenario_4 = {}
# best_models_scenario_4 = {}
# predictions_scenario_4 = {}

# # Initialize dictionaries to store results for the current category
# predictions_scenario_4_category = {}
# metrics_scenario_4_category = {}

# # Before fitting the model, ensure that X_label_onehot_train_scaled and y_train_scaled are of the same length
# assert len(X_label_onehot_train_scaled_4) == len(y_train_scaled_4), "Mismatch in the length of X_label_onehot_train_scaled and y_train_scaled"

# # Iterate over each model, perform GridSearchCV, and make predictions
# for name, model in models.items():
#     print(f"Training and predicting with {name}...")

#     # Perform GridSearchCV to find the best parameters
#     grid_search_4 = GridSearchCV(estimator=model, param_grid=params[name], cv=5, n_jobs=-1, scoring='r2')
#     grid_search_4.fit(X_label_onehot_train_scaled_4, y_train_scaled_4)
    
#     # Store the best model
#     best_models_scenario_4[name] = grid_search_4.best_estimator_
    
#     # Make predictions on the test set
#     y_pred_scaled_4 = grid_search_4.predict(X_label_onehot_test_scaled_4)
        
#     # Inverse transform predictions back to original scale
#     y_pred_original_4 = scaler_y_4.inverse_transform(y_pred_scaled_4.reshape(-1, 1)).flatten()
#     y_test_original_4 = scaler_y_4.inverse_transform(y_test_scaled_4.reshape(-1, 1)).flatten()

#     predictions_scenario_4[name] = y_pred_original_4

#     # Calculate evaluation metrics
#     mse_scenario_4 = mean_squared_error(y_test_original_4, y_pred_original_4)
#     mae_scenario_4 = mean_absolute_error(y_test_original_4, y_pred_original_4)
#     r2_scenario_4 = r2_score(y_test_original_4, y_pred_original_4)
#     rmse_scenario_4 = np.sqrt(mse_scenario_4)
    
#     # Store the metrics
#     metrics_scenario_4[name] = {
#         'Algorithm' : name ,
#         'CrimeCategory': X_test_categories['CrimeCategory'],
#         'ProvinceCode': X_test_categories['ProvinceCode'],
#         'PoliceStationCode': X_test_categories['PoliceStationCode'],
#         'Quarter': X_test_categories['Quarter'],
#         'MSE': mse_scenario_4,
#         'MAE': mae_scenario_4,
#         'R²': r2_scenario_4,
#         'RMSE': rmse_scenario_4
#     }

#     metrics_scenario_4_df = pd.DataFrame(metrics_scenario_4)

#     # Display the metrics
#     print(f"Metrics Scenario 4 for {name}:")
#     print(f"  MSE: {mse_scenario_4}")
#     print(f"  MAE: {mae_scenario_4}")
#     print(f"  R²: {r2_scenario_4}")
#     print(f"  RMSE: {rmse_scenario_4}")
#     print("\n" + "-"*50 + "\n")

#     # Display predicted and true values for comparison
#     print(f"True vs Predicted Values for {name}:")
#     for i in range(len(y_test_original_4)):
#         print(f"True: {y_test_original_4[i]:.4f}, Predicted: {y_pred_original_4[i]:.4f}")

#     print("\n" + "="*50 + "\n")

#     predicted_values_4 = predictions_scenario_4[name] 

#     # Ensure 'predicted_values' is a list or a Series before creating the DataFrame
#     predicted_values_4 = pd.Series(predicted_values_4)

#     # Now create the DataFrame with all columns properly aligned
#     results_4_df = pd.DataFrame({
#         'Algorithm' : name ,
#         'CrimeCategory': X_test_categories['CrimeCategory'],
#         'ProvinceCode': X_test_categories['ProvinceCode'],
#         'PoliceStationCode': X_test_categories['PoliceStationCode'],
#         'Quarter': X_test_categories['Quarter'],
#         'TrueValue': y_test_scaled_4.flatten(),
#         'PredictedValue': predicted_values_4
#     })

#     # Store predictions and metrics for the current category
#     # predictions_per_category_scenario_4[category] = predictions_scenario_3_category
#     # metrics_per_category_scenario_4[category] = metrics_scenario_3_category

#     # Store predictions and metrics for the current category
#     # predictions_per_category_scenario_4[category] = predictions_scenario_4_category
#     # metrics_per_category_scenario_3[category] = metrics_scenario_4_category

#     # ===================================================================================================
# with SplitScalerTab7:
#     with st.expander('Train and test split', expanded=False):
#         st.header(f'Input data for {algorithm} algorithm', divider='rainbow')
#         train_ratio = parameter_split_size
#         test_ratio = 100 - parameter_split_size
#         split_ration_value = f'{train_ratio} : {test_ratio}'
#         split_ration = f'Split Ration % Train\:Test'
        
#         col = st.columns(5)
#         col[0].metric(label="No. of samples", value=X.shape[0], delta="")
#         col[1].metric(label="No. of X variables", value=X.shape[1], delta="")
#         col[2].metric(label="No. of Training samples", value=X_train.shape[0], delta="")
#         col[3].metric(label="No. of Test samples", value=X_test.shape[0], delta="")
#         col[4].metric(label= split_ration, value= split_ration_value, delta="")

    
#     # Display the updated train and test splits
#     with st.expander('Train split : MinMaxScaler', expanded=False):
#         train_col = st.columns((3, 1))
#         with train_col[0]:
#             st.markdown('X_train')
#             st.dataframe(X_train_display, height=210, hide_index=True, use_container_width=True)
#         with train_col[1]:
#             st.markdown('y_train')
#             st.dataframe(y_train_scaled, height=210, hide_index=True, use_container_width=True)

#     with st.expander('Test split : MinMaxScaler', expanded=False):
#         test_col = st.columns((3, 1))
#         with test_col[0]:
#             st.markdown('X_test')
#             st.dataframe(X_test_display, height=210, hide_index=True, use_container_width=True)
#         with test_col[1]:
#             st.markdown('y_test')
#             st.dataframe(y_test_scaled, height=210, hide_index=True, use_container_width=True)

# with EncodingTab8:
#     with st.expander(f'Predictions: Interger Encoding', expanded=False):
#         st.header(f'Train :', divider='rainbow')
#         st.dataframe(output_data, height=210, hide_index=True, use_container_width=True)
#         st.header(f'Test :', divider='rainbow')
        # st.dataframe(data_label_encoded_test, height=210, hide_index=True, use_container_width=True)

#     with st.expander('Label encoding Data : PoliceStationCode and Quarter', expanded=False):
#         st.header(f'Label encoding X_Train', divider='rainbow')
#         st.dataframe(X_train_scaled_1, height=210, hide_index=True, use_container_width=True)
#         st.header(f'Label encoding X_Test', divider='rainbow')
#         st.dataframe(X_test_scaled_1, height=210, hide_index=True, use_container_width=True)
#         st.header(f'Label encoding y_Train', divider='rainbow')
#         st.dataframe(y_train_scaled_1 , height=210, hide_index=True, use_container_width=True)
#         st.header(f'Label encoding y_Test', divider='rainbow')
#         st.dataframe(y_test_scaled_1, height=210, hide_index=True, use_container_width=True)
  
    # with st.expander('One-hot encoding Data : PoliceStationCode and Quarter', expanded=False):
    #     st.header(f'One-hot encoding Train', divider='rainbow')
    #     st.dataframe(data_onehot_encoded_train, height=210, hide_index=True, use_container_width=True)
    #     st.header(f'One-hot encoding Test', divider='rainbow')
    #     st.dataframe(data_onehot_encoded_test, height=210, hide_index=True, use_container_width=True)

    # with st.expander('Label encoding Data : PoliceStationCode and One-hot encoding : Quarter', expanded=False):
    #     st.header(f'Label encoding Train : One-hot encoding', divider='rainbow')
    #     st.dataframe(data_label_onehot_encoded_train, height=210, hide_index=True, use_container_width=True)
    #     st.header(f'Label encoding Test : One-hot encoding', divider='rainbow')
    #     st.dataframe(data_label_onehot_encoded_test, height=210, hide_index=True, use_container_width=True)

    # with st.expander('One-hot encoding Data : PoliceStationCode and Label encoding : Quarter', expanded=False):
    #     st.header(f'Label encoding Train : One-hot encoding', divider='rainbow')
    #     st.dataframe(data_onehot_label_encoded_train, height=210, hide_index=True, use_container_width=True)
    #     st.header(f'Label encoding Test : One-hot encoding', divider='rainbow')
    #     st.dataframe(data_onehot_label_encoded_test, height=210, hide_index=True, use_container_width=True)

with RealTimePreditionsTab9:
    with st.expander(f'Predictions : Without Encoding', expanded=False):
        st.header(f'Predictions and True values :', divider='rainbow')
        st.dataframe(output_data, height=210, hide_index=True, use_container_width=True)
    
#     with st.expander(f'Predictions : Label encoding : PoliceStationCode and Quarter', expanded=False):
#         st.header(f'Train :', divider='rainbow')
#         st.dataframe(results_1_df, height=210, hide_index=True, use_container_width=True)
#         st.header(f'Test :', divider='rainbow')
#         # st.dataframe(data_label_encoded_test, height=210, hide_index=True, use_container_width=True)

#     with st.expander('Predictions : One-hot encoding Predictions : PoliceStationCode and Quarter', expanded=False):
#         st.header(f'Train :', divider='rainbow')
#         st.dataframe(results_2_df, height=210, hide_index=True, use_container_width=True)
#         st.header(f'Test :', divider='rainbow')
#         # st.dataframe(data_onehot_encoded_test, height=210, hide_index=True, use_container_width=True)

#     with st.expander('Predictions : Label encoding : PoliceStationCode and One-hot encoding : Quarter', expanded=False):
#         st.header(f'Train :', divider='rainbow')
#         st.dataframe(results_3_df, height=210, hide_index=True, use_container_width=True)
#         st.header(f'Test :', divider='rainbow')
#         # st.dataframe(data_label_onehot_encoded_test, height=210, hide_index=True, use_container_width=True) 
         
#     with st.expander('Predictions : One-hot encoding: PoliceStationCode and Label encoding : Quarter', expanded=False):
#         st.header(f'Train :', divider='rainbow')
#         st.dataframe(results_4_df, height=210, hide_index=True, use_container_width=True)
#         st.header(f'Test :', divider='rainbow')
#         # st.dataframe(data_onehot_label_encoded_test, height=210, hide_index=True, use_container_width=True)  

with RealTimeMetricsTab10:
    with st.expander(f'Mertics: Without Encoding', expanded=False):
        st.header(f'Train :', divider='rainbow')
        st.dataframe(output_mertics, height=210, hide_index=True, use_container_width=True)
        st.header(f'Test :', divider='rainbow')
        # st.dataframe(data_label_encoded_test, height=210, hide_index=True, use_container_width=True)

#     with st.expander(f'Metrics : Label encoding : PoliceStationCode and Quarter', expanded=False):
#         st.header(f'Train :', divider='rainbow')
#         st.dataframe(metrics_scenario_1_df, height=210, hide_index=True, use_container_width=True)
#         st.header(f'Test :', divider='rainbow')
#         # st.dataframe(data_label_encoded_test, height=210, hide_index=True, use_container_width=True)

#     with st.expander('Mertics: One-hot encoding Predictions : PoliceStationCode and Quarter', expanded=False):
#         st.header(f'Train :', divider='rainbow')
#         st.dataframe(metrics_scenario_2_df, height=210, hide_index=True, use_container_width=True)
#         st.header(f'Test :', divider='rainbow')
#         # st.dataframe(data_onehot_encoded_test, height=210, hide_index=True, use_container_width=True)

#     with st.expander('Mertics : Label encoding : PoliceStationCode and One-hot encoding : Quarter', expanded=False):
#         st.header(f'Train :', divider='rainbow')
#         st.dataframe(metrics_scenario_3_df, height=210, hide_index=True, use_container_width=True)
#         st.header(f'Test :', divider='rainbow')
#         # st.dataframe(data_label_onehot_encoded_test, height=210, hide_index=True, use_container_width=True) 
         
#     with st.expander('Mertics : One-hot encoding: PoliceStationCode and Label encoding : Quarter', expanded=False):
#         st.header(f'Train :', divider='rainbow')
#         st.dataframe(metrics_scenario_4_df, height=210, hide_index=True, use_container_width=True)
#         st.header(f'Test :', divider='rainbow')
#         # st.dataframe(data_onehot_label_encoded_test, height=210, hide_index=True, use_container_width=True) 
    

