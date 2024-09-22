import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split
    )
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
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
from modelTransformationResponse.outliersResponse import (
    identify_outliers_data,
    replace_outliers_data
    )
from sklearn.metrics import (
    mean_absolute_error ,
    mean_squared_error ,
    r2_score,
    mean_absolute_percentage_error,
    adjusted_rand_score
)
from math import sqrt
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

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

            st.markdown('**1.4. Set Test and Train Parameters**')
            parameter_split_size = st.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

            st.subheader('2. Select Algorithm')
            with st.expander('Algorithms'):
                algorithm = st.radio('', options=['ANN (MLPRegressor)', 'KNN', 'RFM', 'SVR','XGBoost'], index=2)

            st.subheader('3. Learning Parameters')
            with st.expander('See parameters', expanded=False):

                # if algorithm in ['RFM' , 'XGBoost']:
                #     parameter_n_estimators = st.slider('Number of estimators (n_estimators)', 10, 50, 100)  #1000
                
                # if algorithm in ['RFM' , 'XGBoost']:
                #     # st.subheader('4. General Parameters')
                #     # with st.expander('See parameters', expanded=False):
                #     parameter_random_state = st.slider('Seed number (random_state)', 0, 1000, 42, 1)

                # if algorithm == 'RFM':
                #         parameter_max_features = st.select_slider('Max features (max_features)', options=['sqrt', 'log2'])
                #         parameter_min_samples_split = st.slider('Minimum number of samples required to split an internal node (min_samples_split)', 1, 2, 5, 10)
                #         parameter_min_samples_leaf = st.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)
                #         parameter_criterion = st.select_slider('Performance measure (criterion)', options=['squared_error', 'absolute_error', 'friedman_mse', 'poisson'])
                #         parameter_bootstrap = st.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
                #         parameter_oob_score = st.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])
                # elif algorithm == 'ANN (MLPRegressor)':
                #         parameter_hidden_layer_size = st.select_slider('Hidden layers size is the  number of neorons in each hidden layer (hidden_layer_size)', options=[(50, 50), (100,)])
                #         parameter_solver = st.select_slider('Solver for weight optimization (solver) ', options=['adam', 'sgd'])
                #         parameter_activation = st.select_slider('Activation function for the hidden layer (activation)', options=['tanh', 'relu'])
                # elif algorithm == 'KNN':
                #         parameter_n_neighbors = st.select_slider('Number of neighbors to use (n_neighbors )', options= [3, 5, 7])
                #         parameter_weights = st.select_slider('Weight function used in prediction (weights)', options=['uniform', 'distance'])
                # elif algorithm == 'SVR':
                #         parameter_kernel = st.select_slider('Specifies the kernel type to be (kernel)', options=['linear', 'rbf'])
                #         parameter_C = st.select_slider('Regularization parameter (C)', options=[1,10])
                #         parameter_epsilon = st.select_slider('Epsilon in the epsilon-SVR (epsilon)', options=[0.1 , 0.2])
                # elif algorithm == 'XGBoost':
                #         parameter_learning_rate = st.select_slider('Boosting learning rate (learning_rate)', options=[0.01, 0.1])
                #         parameter_max_depth = st.select_slider('Maximum depth of a tree (max_depth)', options=[3,5,7])
                #         parameter_min_child_weight = st.select_slider('Minimum sum of instance weight (hessian) needed in a child (min_child_weight)', options=[1,3])
                #         parameter_subsample = st.select_slider('Subsample ratio of the training instances (subsample)', options=[0.8, 1.0])
                #         parameter_cosample_bytree = st.select_slider('Subsample ratio of columns when constructing each tree (cosample_bytree)', options=[0.8, 1.0])

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
        df_replace_outliers = replace_outliers_data(df_suggeted_province_quarterly_data_db)
    
        # Display the DataFrame sorted by 'PoliceStationCode'
        st.dataframe(df_replace_outliers.sort_values(by='PoliceStationCode'), height=210, use_container_width=True)

        # Melt the DataFrame for visualization purposes
        df_outliers_replaced_melt = df_replace_outliers.melt(
            id_vars=['CrimeCategory', 'ProvinceCode', 'PoliceStationCode', 'Quarter'],
            var_name='Year',
            value_name='Percentage'
        )

        # Convert 'Percentage' to numeric if it's not already
        df_outliers_replaced_melt['Percentage'] = pd.to_numeric(df_outliers_replaced_melt['Percentage'], errors='coerce')

        # Create the line plot
        plt.figure(figsize=(12, 8))
        sns.lineplot(data=df_outliers_replaced_melt, x='Year', y='Percentage', hue='CrimeCategory', marker='o')
    
        # Add title and labels
        plt.title('Crime Trends Over the Years with Outliers Replaced')
        plt.xlabel('Year')
        plt.ylabel('Percentage')

        # Adjust legend placement
        plt.legend(title='Crime Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)

        # Annotate each data point with its value
        for i in range(df_outliers_replaced_melt.shape[0]):
            plt.text(
                x=i % len(df_outliers_replaced_melt['Year'].unique()),  # Corrected x-coordinate for annotation
                y=df_outliers_replaced_melt['Percentage'].iloc[i],
                s=f"{df_outliers_replaced_melt['Percentage'].iloc[i]:.2f}",
                color='black',
                ha='right',
                va='bottom'
            )

        # Adjust layout to make room for the legend
        plt.tight_layout()

        # Display the graph in Streamlit
        st.pyplot(plt)


# Initialize models and parameter grid
models = {
    'RFM': RandomForestRegressor(),
    'SVR': SVR(),
    'XGBR': XGBRegressor(),
    'KNNR': KNeighborsRegressor(),
    'MLPR': MLPRegressor(2000)
}

params = {
    'RFM': {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]},
    'SVR': {'C': [1, 10], 'kernel': ['linear', 'rbf']},
    'XGBR': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]},
    'KNNR': {'n_neighbors': [5, 10], 'weights': ['uniform', 'distance']},
    'MLPR': {
        'hidden_layer_sizes': [(100,), (100, 50)],
        'activation': ['relu', 'tanh'],
        'learning_rate_init': [0.0001, 0.001, 0.01],
        'max_iter': [500, 1000, 2000],  # Increase max_iter to allow more iterations
        'solver': ['adam', 'lbfgs']
    }
}

def evaluate_metrics(y_true, y_pred):
        metrics_list = {
        'Prediction': y_pred,
        'True_value': y_test,
        'MSE': mean_squared_error(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'R²': r2_score(y_test, y_pred),
        'MAPE': mean_absolute_percentage_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
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
df_cleaned = df_replace_outliers.copy()

# Assuming df is your DataFrame
def replace_non_finite_with_median(df):
    # Replace inf, -inf with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Replace NaN with median
    df = df.apply(lambda x: x.fillna(x.median()) if np.issubdtype(x.dtype, np.number) else x)
    
    return df

# Example usage:
df_wide = replace_non_finite_with_median(df_cleaned)

# B - Defining feature set X (years 2016-2023) and target y (2024 prediction) =====
# We exclude 'CrimeCategory', 'ProvinceCode', 'PoliceStationCode', 'Quarter' as these are categorical
X = df_wide[['2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']]
y = df_wide['2023']  # Using 2023 as a proxy for training, predicting 2024

# C - Splitting the dataset into train/test sets (use 2016-2023 data for training) =====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensure X_train and X_test are DataFrames before scaling
X_train_display = pd.DataFrame(X_train, columns=X.columns)
X_train_display = pd.DataFrame(X_test, columns=X.columns)

# D - Scaling the numeric features ===== Use values for Predicting withou Encoding
scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# predictions = {}

# # Loop through each model and perform GridSearchCV
# for name, model in models.items():
#     print(f"Training {name} model...")
    
#     # Get the parameter grid for the current model
#     param_grid = params[name]
    
#     # Define GridSearchCV
#     grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    
#     # Fit the model using GridSearchCV
#     grid_search.fit(X_train_scaled, y_train)
    
#     # Retrieve the best estimator (model with best parameters)
#     best_model = grid_search.best_estimator_
    
#     # Predict on the test set
#     y_pred = best_model.predict(X_test_scaled)
    
#     # Store predictions and calculate metrics
#     predictions[name] = {
#         'Prediction': y_pred,
#         'True_value': y_test,
#         'MSE': mean_squared_error(y_test, y_pred),
#         'MAE': mean_absolute_error(y_test, y_pred),
#         'R²': r2_score(y_test, y_pred),
#         'MAPE': mean_absolute_percentage_error(y_test, y_pred),
#         'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
#     }

# # F - Merging predictions with original dataset =====
# # Create a DataFrame for the output format
# output_data = df_wide[['CrimeCategory', 'ProvinceCode', 'PoliceStationCode', 'Quarter', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']].copy()

# output_mertics = df_wide[['CrimeCategory', 'ProvinceCode', 'PoliceStationCode', 'Quarter', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']].copy()

# # Adding predictions and true values for each model
# for name, results in predictions.items():
#     # Adding prediction for 2024 as a new column
#     output_data[f'Prediction_{name}'] = pd.Series(results['Prediction'], index=y_test.index)
    
#     # Adding true values for 2023 (actual values from the test set)
#     output_data[f'True_value_{name}'] = pd.Series(y_test.values, index=y_test.index)

#     output_mertics[f'MSE_{name}'] = pd.Series(results['MSE'], index=y_test.index)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Function to replace non-finite and NaN values with the median
def replace_non_finite_with_median(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.apply(lambda x: x.fillna(x.median()) if np.issubdtype(x.dtype, np.number) else x)
    return df

# Scenario functions
def scenario_1(df):
    label_encoder_psc = LabelEncoder()
    label_encoder_qtr = LabelEncoder()
    df['PoliceStationCode'] = label_encoder_psc.fit_transform(df['PoliceStationCode'])
    df['Quarter'] = label_encoder_qtr.fit_transform(df['Quarter'])
    return df

def scenario_2(df):
    onehot_encoder = OneHotEncoder(sparse_output=False)
    encoded_features = onehot_encoder.fit_transform(df[['PoliceStationCode', 'Quarter']])
    encoded_df = pd.DataFrame(encoded_features, columns=onehot_encoder.get_feature_names_out(['PoliceStationCode', 'Quarter']))
    df = pd.concat([df, encoded_df], axis=1).drop(['PoliceStationCode', 'Quarter'], axis=1)
    return df

def scenario_3(df):
    label_encoder_psc = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse_output=False)
    df['PoliceStationCode'] = label_encoder_psc.fit_transform(df['PoliceStationCode'])
    encoded_features = onehot_encoder.fit_transform(df[['Quarter']])
    encoded_df = pd.DataFrame(encoded_features, columns=onehot_encoder.get_feature_names_out(['Quarter']))
    df = pd.concat([df, encoded_df], axis=1).drop(['Quarter'], axis=1)
    return df

def scenario_4(df):
    label_encoder_qtr = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse_output=False)
    df['Quarter'] = label_encoder_qtr.fit_transform(df['Quarter'])
    encoded_features = onehot_encoder.fit_transform(df[['PoliceStationCode']])
    encoded_df = pd.DataFrame(encoded_features, columns=onehot_encoder.get_feature_names_out(['PoliceStationCode']))
    df = pd.concat([df, encoded_df], axis=1).drop(['PoliceStationCode'], axis=1)
    return df

# Train and predict function
def train_and_predict(df, models, params, scenario_func):
    df_encoded = scenario_func(df.copy())
    
    # Adjust the feature set to include the encoded columns as necessary
    features = [col for col in df_encoded.columns if col not in ['CrimeCategory', 'ProvinceCode', 'Quarter', '2023']]
    
    X = df_encoded[features]
    y = df_encoded['2023']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    predictions = []
    metrics = []

    for name, model in models.items():
        grid_search = GridSearchCV(model, params[name], cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test_scaled)

        # Collect predictions
        prediction_result = {
            'Model': name,
            'Prediction': y_pred,
            'True_value': y_test.values
        }
        predictions.append(prediction_result)

        # Calculate metrics
        metric_result = {
            'Model': name,
            'MAE': mean_absolute_error(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'R²': r2_score(y_test, y_pred),
            'MAPE': mean_absolute_percentage_error(y_test, y_pred),
            'ARS': adjusted_rand_score(y_test, y_pred)  # Though typically for clustering, included as requested
        }
        metrics.append(metric_result)

    return pd.DataFrame(predictions), pd.DataFrame(metrics)

# Run scenarios
scenario_funcs = [scenario_1, scenario_2, scenario_3, scenario_4]
all_predictions = []
all_metrics = []

for scenario_func in scenario_funcs:
    scenario_predictions, scenario_metrics = train_and_predict(df_cleaned, models, params, scenario_func)
    all_predictions.append(scenario_predictions)
    all_metrics.append(scenario_metrics)

# Concatenate all predictions and metrics
final_predictions = pd.concat(all_predictions, axis=0).reset_index(drop=True)
final_metrics = pd.concat(all_metrics, axis=0).reset_index(drop=True)

# Add categorical columns for reference
final_predictions = pd.concat([df_cleaned[['CrimeCategory', 'ProvinceCode', 'PoliceStationCode', 'Quarter']].reset_index(drop=True), final_predictions], axis=1)
final_metrics = pd.concat([df_cleaned[['CrimeCategory', 'ProvinceCode', 'PoliceStationCode', 'Quarter']].reset_index(drop=True), final_metrics], axis=1)

# Display predictions and metrics in Streamlit
st.dataframe(final_predictions, height=300, use_container_width=True)
st.dataframe(final_metrics, height=300, use_container_width=True)
                   

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

with RealTimePreditionsTab9:
    with st.expander(f'Predictions : Without Encoding', expanded=False):
        st.header(f'Predictions and True values :', divider='rainbow')
        st.dataframe(df_cleaned, height=210, hide_index=True, use_container_width=True)

    with st.expander(f'Predictions : With Encoding', expanded=False):
        st.header(f'Predictions and True values :', divider='rainbow')
        st.dataframe(df_wide, height=210, hide_index=True, use_container_width=True)

with RealTimeMetricsTab10:
    with st.expander(f'Mertics: Without Encoding', expanded=False):
        st.header(f'Train :', divider='rainbow')
        st.dataframe(predictions, height=210, hide_index=True, use_container_width=True)

    with st.expander(f'Mertics: Encoding', expanded=False):
        st.header(f'Train :', divider='rainbow')
        st.dataframe(final_predictions, height=210, hide_index=True, use_container_width=True)
        st.header(f'Test :', divider='rainbow')
        st.dataframe(final_metrics, height=210, hide_index=True, use_container_width=True)


