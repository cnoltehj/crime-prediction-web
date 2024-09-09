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
    mean_absolute_percentage_error
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

AboutTab1, DataTab2, TransformTab3, SplitScalerTab4, EncodeTab5, RFMPredictionTab6, SVRPredictionTab7, XGBoostPredictionTab8, ANNPredictionTab9, KNNPredictionTab10, AdHocTab11 = st.tabs(['About','Data','Transformation','Scaler-Split', 'Encoding','RFM-Preditions','SVR-Preditions','XGBoost-Preditions','ANN-Preditions','KNN-Preditions','AdHoc-Shapley'])

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
with DataTab2:
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

        #if visualise_initail_median:
with TransformTab3:
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
        'XGBR': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]},
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

    # Identify year columns dynamically
    year_columns = [col for col in df_replace_outliers.columns if col.isdigit() and len(col) == 4]

    # C - Splitting the Data =========
    # C - Splitting the Data =========

    # Define the feature set X and target variable y
    X = df_replace_outliers.drop(columns=['CrimeCategory', 'ProvinceCode', 'PoliceStationCode', 'Quarter'])
    y = df_replace_outliers[year_columns].median(axis=1)  # Median across all years

    # Extract the required columns for display
    crime_category = df_replace_outliers['CrimeCategory']
    province_code = df_replace_outliers['ProvinceCode']
    police_station_code = df_replace_outliers['PoliceStationCode']
    quarter = df_replace_outliers['Quarter']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test, crime_category_train, crime_category_test, \
    province_code_train, province_code_test, police_station_code_train, police_station_code_test, \
    quarter_train, quarter_test = train_test_split(
        X, y, crime_category, province_code, police_station_code, quarter,
        test_size=(100 - parameter_split_size) / 100, random_state=42
    )

    # B - Scaling with MinMaxScaler =========
    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Apply MinMaxScaler to the feature set (X) in both training and testing sets
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Reshape y_train and y_test for scaling
    y_train_reshaped = y_train.values.reshape(-1, 1)
    y_test_reshaped = y_test.values.reshape(-1, 1)

    # Apply MinMaxScaler to y_train and y_test
    y_train_scaled = scaler.fit_transform(y_train_reshaped)
    y_test_scaled = scaler.transform(y_test_reshaped)   

    # Convert back to 1D arrays (if needed)
    y_train_scaled = y_train_scaled.flatten()
    y_test_scaled = y_test_scaled.flatten()

    # Convert the scaled arrays back to dataframes with the same column names as X
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # Reset indices for the extracted columns to ensure proper alignment
    crime_category_train = crime_category_train.reset_index(drop=True)
    crime_category_test = crime_category_test.reset_index(drop=True)
    province_code_train = province_code_train.reset_index(drop=True)
    province_code_test = province_code_test.reset_index(drop=True)
    police_station_code_train = police_station_code_train.reset_index(drop=True)
    police_station_code_test = police_station_code_test.reset_index(drop=True)
    quarter_train = quarter_train.reset_index(drop=True)
    quarter_test = quarter_test.reset_index(drop=True)


    for name, model in models.items():

        grid_search_model_train = GridSearchCV(model, params[name], cv=cv_value, scoring='neg_mean_squared_error')
            #GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring_value, cv=cv_value,error_score='raise')
        grid_search_model_train.fit(X_train, y_train_scaled)
            # Get the best model from GridSearchCV
        best_model_train = grid_search_model_train.best_estimator_

        grid_search_model_test = GridSearchCV(model, params[name], cv=cv_value, scoring='neg_mean_squared_error')
            #GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring_value, cv=cv_value,error_score='raise')
        grid_search_model_test.fit(X_test, y_test_scaled)
            # Get the best model from GridSearchCV
        best_model_test = grid_search_model_train.best_estimator_

    # E -  Prediction and Metrics =========

        for name, model in best_estimators_train.items():
            y_train_pred = model.predict(X_train)

            print(f'Train Prediction without encoding for {name}: {y_train_pred}')
        
            mse = mean_squared_error(y_train_scaled, y_train_pred)
            mae = mean_absolute_error(y_train_scaled, y_train_pred)
            r2 = r2_score(y_train_scaled, y_train_pred)
            mape= mean_absolute_percentage_error(y_train_scaled, y_train_pred)
            rmse = np.sqrt(mse) 

            metrics[name] = {'MSE': mse, 'MAE': mae, 'R²': r2, 'RMSE': rmse}

            print(f'Metrics Train without encoding  for {name}: {metrics[name]}')

        for name, model in best_estimators_test.items():
            y_test_pred = model.predict(X_test)

            print(f'Test Prediction without encoding  for {name}: {y_test_pred}')
        
            mse = mean_squared_error(y_test_scaled, y_test_pred)
            mae = mean_absolute_error(y_test_scaled, y_test_pred)
            r2 = r2_score(y_test_scaled, y_test_pred)
            mape= mean_absolute_percentage_error(y_test_scaled, y_test_pred)
            rmse = np.sqrt(mse) 

            metrics[name] = {'MSE': mse, 'MAE': mae, 'R²': r2, 'RMSE': rmse}

            print(f'Metrics Test without encoding for {name}: {metrics[name]}')

    # F - Insert additional columns for display purposes =========

    # Insert additional columns for display purposes
    X_train_display = X_train_scaled_df.copy()
    X_train_display.insert(0, 'CrimeCategory', crime_category_train)
    X_train_display.insert(1, 'ProvinceCode', province_code_train)
    X_train_display.insert(2, 'PoliceStationCode', police_station_code_train)
    X_train_display.insert(3, 'Quarter', quarter_train)

    X_test_display = X_test_scaled_df.copy()
    X_test_display.insert(0, 'CrimeCategory', crime_category_test)
    X_test_display.insert(1, 'ProvinceCode', province_code_test)
    X_test_display.insert(2, 'PoliceStationCode', police_station_code_test)
    X_test_display.insert(3, 'Quarter', quarter_test)

    # G - Train dataset Encoding Scenarios and Predictions =========

    # Scenario 1 - Train: Label encoding for 'PoliceStationCode' and 'Quarter'
    label_encoder_psc_train = LabelEncoder()
    label_encoder_qtr_train = LabelEncoder()
    onehot_encoder_train = OneHotEncoder()

    # Copy the training data
    data_label_encoded_train = X_train_display.copy()
    # Encode 'PoliceStationCode' and 'Quarter' in the training set
    data_label_encoded_train['PoliceStationCode'] = label_encoder_psc_train.fit_transform(data_label_encoded_train['PoliceStationCode'])
    data_label_encoded_train['Quarter'] = label_encoder_qtr_train.fit_transform(data_label_encoded_train['Quarter'])

    # Apply the same encoding to the test set
    data_label_encoded_test = X_test_display.copy()
    data_label_encoded_test['PoliceStationCode'] = label_encoder_psc_train.transform(data_label_encoded_test['PoliceStationCode'])
    data_label_encoded_test['Quarter'] = label_encoder_qtr_train.transform(data_label_encoded_test['Quarter'])

    # Select only numerical columns
    numerical_columns = data_label_encoded_train.select_dtypes(include=[np.number]).columns
    data_numerical_only_train = data_label_encoded_train[numerical_columns]
    data_numerical_only_test = data_label_encoded_test[numerical_columns]

    # Apply StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(data_numerical_only_train)
    X_test_scaled = scaler.transform(data_numerical_only_test)

    # Train and Evaluate
    metrics_scenario_1 = {}
    best_models_scenario_1 = {}
    predictions_scenario_1 = {}

      # Before fitting the model, ensure that X_train_scaled and y_train_scaled are of the same length
    assert len(X_train_scaled) == len(y_train_scaled), "Mismatch in the length of X_train_scaled and y_train_scaled"

   # Initialize dictionaries to store predictions and metrics for each category
    predictions_per_category = {}
    metrics_per_category = {}


    print("Df_Replace_Outliers")
    print(df_replace_outliers)
    print("X Train Scaled DF")
    print(X_train_scaled_df)
    print("Y Train")
    print(y_train)

    print("X Test Scaled DF")
    print(X_test_scaled_df)
    print("Y Test")
    print(y_test)

  
# Iterate over each unique CrimeCategory in the dataset
for category in df_replace_outliers['CrimeCategory'].unique():
    print(f"Processing CrimeCategory: {category}")
    
    # Filter the dataset for the current category
    X_category = X_train_scaled_df[df_replace_outliers['CrimeCategory'] == category]
    
    # Drop the specified columns if they exist in the DataFrame
    columns_to_drop = ['CrimeCategory', 'ProvinceCode', 'PoliceStationCode', 'Quarter']
    X_category = X_category.drop(columns=[col for col in columns_to_drop if col in X_category.columns])
    
    # Filter the target variable
    y_category = y_train[df_replace_outliers['CrimeCategory'] == category]

    
    # Check if the lengths of X_category and y_category are consistent
    if len(X_category) != len(y_category):
        print(f"Skipping category {category} due to inconsistent sample sizes.")
        continue
    
    # Check if there's enough data to train the model
    if len(X_category) < 2 or len(y_category) < 2:
        print(f"Skipping category {category} due to insufficient data.")
        continue
    
    # Initialize dictionaries to store results for the current category
    predictions_scenario_1_category = {}
    metrics_scenario_1_category = {}
    
    # Iterate over each model, perform GridSearchCV, and make predictions
    for name, model in models.items():
        print(f"Training and predicting with {name} for {category}...")

         # For KNeighborsRegressor, adjust n_neighbors based on the available samples
        if name == 'KNNR':
            n_samples_fit = len(X_category)
            if n_samples_fit < 5:  # If fewer than 5 samples, adjust n_neighbors to n_samples_fit
                params['KNNR']['n_neighbors'] = [n_samples_fit]
        
        # Perform GridSearchCV to find the best parameters
        grid_search = GridSearchCV(estimator=model, param_grid=params[name], cv=2, n_jobs=-1, scoring='r2')
        grid_search.fit(X_category, y_category)
        
        # Store the best model
        best_models_scenario_1[name] = grid_search.best_estimator_

        # Prepare X_test_scaled for prediction by excluding the categorical columns
        columns_to_drop = ['CrimeCategory', 'ProvinceCode', 'PoliceStationCode', 'Quarter']

        # Prepare X_test_scaled for prediction by excluding the categorical columns
        # Only drop columns if they exist in X_test_scaled_df
        X_test_scaled_filtered = X_test_scaled_df.drop(columns=[col for col in columns_to_drop if col in X_test_scaled_df.columns])

        # X_test_scaled_filtered = X_test_scaled_df.drop(['CrimeCategory', 'ProvinceCode', 'PoliceStationCode', 'Quarter'], axis=1)
        
        # Make predictions on the test set
        y_pred = grid_search.predict(X_test_scaled_filtered)
        predictions_scenario_1[name] = y_pred
        
        # Calculate evaluation metrics
        mse_scenario_1 = mean_squared_error(y_test_scaled, y_pred)
        mae_scenario_1 = mean_absolute_error(y_test_scaled, y_pred)
        r2_scenario_1 = r2_score(y_test_scaled, y_pred)
        rmse_scenario_1 = np.sqrt(mse_scenario_1)
        
        # Store the metrics
        metrics_scenario_1[name] = {
            'MSE': mse_scenario_1,
            'MAE': mae_scenario_1,
            'R²': r2_scenario_1,
            'RMSE': rmse_scenario_1
        }
        
        # Display the metrics
        print(f"Metrics scenario_1 for {name}:")
        print(f"  MSE: {mse_scenario_1}")
        print(f"  MAE: {mae_scenario_1}")
        print(f"  R²: {r2_scenario_1}")
        print(f"  RMSE: {rmse_scenario_1}")
        print("\n" + "-"*50 + "\n")

    
    # Store predictions and metrics for the current category
    predictions_per_category[category] = predictions_scenario_1_category
    metrics_per_category[category] = metrics_scenario_1_category


    # Scenario 2 Train: One-hot encoding for 'PoliceStationCode' and 'Quarter'

    data_onehot_encoded_train = X_train_display.copy()
    encoded_features_onehot_encoded_train = onehot_encoder_train.fit_transform(data_onehot_encoded_train[['PoliceStationCode', 'Quarter']]).toarray()
    encoded_df_onehot_encoded_train = pd.DataFrame(encoded_features_onehot_encoded_train, columns=onehot_encoder_train.get_feature_names_out(['PoliceStationCode', 'Quarter']))
    data_onehot_encoded_train = pd.concat([data_onehot_encoded_train, encoded_df_onehot_encoded_train], axis=1).drop(['PoliceStationCode', 'Quarter'], axis=1)

    # Apply the same encoding to the test set
    data_onehot_encoded_test = X_test_display.copy()
    encoded_features_onehot_encoded_test = onehot_encoder_train.transform(data_onehot_encoded_test[['PoliceStationCode', 'Quarter']]).toarray()
    encoded_df_onehot_encoded_test = pd.DataFrame(encoded_features_onehot_encoded_test, columns=onehot_encoder_train.get_feature_names_out(['PoliceStationCode', 'Quarter']))
    data_onehot_encoded_test = pd.concat([data_onehot_encoded_test, encoded_df_onehot_encoded_test], axis=1).drop(['PoliceStationCode', 'Quarter'], axis=1)

    # Select only numerical columns
    numerical_columns_onehot_encoded = data_onehot_encoded_train.select_dtypes(include=[np.number]).columns
    data_numerical_onehot_only_train = data_onehot_encoded_train[numerical_columns_onehot_encoded]
    data_numerical_onehot_only_test = data_onehot_encoded_test[numerical_columns_onehot_encoded]

    # Apply StandardScaler
    scaler = StandardScaler()
    X_onehot_train_scaled = scaler.fit_transform(data_numerical_onehot_only_train)
    X_onehot_test_scaled = scaler.transform(data_numerical_onehot_only_test)

    # Train and Evaluate
    metrics_scenario_2 = {}
    best_models_scenario_2 = {}
    predictions_scenario_2 = {}


    # st.write("X_category") 
    # st.write(X_category) 
    # st.write("y_category") 
    # st.write(y_category) 
    # st.write("y_train_display") 
    # st.write(y_test) 
    # st.write("X_train_display") 
    # st.write(X_test_display) 
    # st.write("X_train_display") 
    # st.write(X_train_display) 
    # st.write("y_train") 
    # st.write(y_train) 
    # st.write("X_train X_onehot_train_scaled")
    # st.write(X_train)
    # st.write("y_train_scaled X_onehot_train_scaled")
    # st.write(y_train_scaled)

    # Before fitting the model, ensure that X_onehot_train_scaled and y_train_scaled are of the same length
    assert len(X_onehot_train_scaled) == len(y_train_scaled), "Mismatch in the length of X_onehot_train_scaled and y_train_scaled"

    # Iterate over each model, perform GridSearchCV, and make predictions
    for name, model in models.items():
        print(f"Training and predicting with {name}...")
        
        # Perform GridSearchCV to find the best parameters
        grid_search = GridSearchCV(estimator=model, param_grid=params[name], cv=5, n_jobs=-1, scoring='r2')
        grid_search.fit(X_onehot_train_scaled, y_train_scaled)
        
        # Store the best model
        best_models_scenario_2[name] = grid_search.best_estimator_
        
        # Make predictions on the test set
        y_pred = grid_search.predict(X_onehot_test_scaled)
        predictions_scenario_2[name] = y_pred
        
        # Calculate evaluation metrics
        mse_scenario_2= mean_squared_error(y_test_scaled, y_pred)
        mae_scenario_2 = mean_absolute_error(y_test_scaled, y_pred)
        r2_scenario_2= r2_score(y_test_scaled, y_pred)
        rmse_scenario_2= np.sqrt(mse_scenario_2)
        
        # Store the metrics
        metrics_scenario_2[name] = {
            'MSE': mse_scenario_2,
            'MAE': mae_scenario_2,
            'R²': r2_scenario_2,
            'RMSE': rmse_scenario_2
        }
        
        # Display the metrics
        print(f"Metrics Scenario_2 for {name}:")
        print(f"  MSE: {mse_scenario_2}")
        print(f"  MAE: {mae_scenario_2}")
        print(f"  R²: {r2_scenario_2}")
        print(f"  RMSE: {rmse_scenario_2}")
        print("\n" + "-"*50 + "\n")

    # Scenario 3 Train: Label encoding for 'PoliceStationCode' and one-hot encoding for 'Quarter'
    data_label_onehot_encoded_train = X_train_display.copy()
    data_label_onehot_encoded_train['PoliceStationCode'] = label_encoder_psc_train.fit_transform(data_label_onehot_encoded_train['PoliceStationCode'])
    encoded_qtr_train = onehot_encoder_train.fit_transform(data_label_onehot_encoded_train[['Quarter']]).toarray()
    encoded_qtr_df_train = pd.DataFrame(encoded_qtr_train, columns=onehot_encoder_train.get_feature_names_out(['Quarter']))
    data_label_onehot_encoded_train = pd.concat([data_label_onehot_encoded_train, encoded_qtr_df_train], axis=1).drop(['Quarter'], axis=1)

    # Apply the same encoding to the test set
    data_label_onehot_encoded_test = X_test_display.copy()
    data_label_onehot_encoded_test['PoliceStationCode'] = label_encoder_psc_train.fit_transform(data_label_onehot_encoded_test['PoliceStationCode'])
    encoded_qtr_test = onehot_encoder_train.fit_transform(data_label_onehot_encoded_test[['Quarter']]).toarray()
    encoded_qtr_df_test = pd.DataFrame(encoded_qtr_test, columns=onehot_encoder_train.get_feature_names_out(['Quarter']))
    data_label_onehot_encoded_test = pd.concat([data_label_onehot_encoded_test, encoded_qtr_df_test], axis=1).drop(['Quarter'], axis=1)

    # # Select only numerical columns
    # numerical_columns_train = data_label_onehot_encoded_train.select_dtypes(include=['float64', 'int64']).columns
    # data_label_onehot_encoded_train = data_label_onehot_encoded_train[numerical_columns_train]
    # data_label_onehot_encoded_test = data_label_onehot_encoded_test[numerical_columns_train]

    # Select only numerical columns
    numerical_columns_train = data_label_onehot_encoded_train.select_dtypes(include=['float64', 'int64']).columns

    # Ensure the test set has the same columns as the train set
    missing_cols = set(numerical_columns_train) - set(data_label_onehot_encoded_test.columns)
    for col in missing_cols:
        data_label_onehot_encoded_test[col] = 0

    # Reorder columns in the test set to match the train set
    data_label_onehot_encoded_test = data_label_onehot_encoded_test[numerical_columns_train]

    # Now you can safely subset the train and test DataFrames
    data_label_onehot_encoded_train = data_label_onehot_encoded_train[numerical_columns_train]


    # Apply StandardScaler
    scaler = StandardScaler()
    data_label_onehot_encoded_train_scaled = scaler.fit_transform(data_label_onehot_encoded_train)
    data_label_onehot_encoded_test_scaled = scaler.transform(data_label_onehot_encoded_test)

    # Train and Evaluate
    metrics_scenario_3 = {}
    best_models_scenario_3 = {}
    predictions_scenario_3 = {}
    metrics_scenario_3 = {}

    for name, model in models.items():
        print(f"Training and predicting with {name}...")
        
        # Perform GridSearchCV to find the best parameters
        grid_search = GridSearchCV(estimator=model, param_grid=params[name], cv=5, n_jobs=-1, scoring='r2')
        grid_search.fit(X_train_scaled, y_train_scaled)
        
        # Store the best model
        best_models_scenario_1[name] = grid_search.best_estimator_
        
        # Make predictions on the test set
        y_pred = grid_search.predict(X_test_scaled)
        predictions_scenario_1[name] = y_pred
        
        # Calculate evaluation metrics
        mse_scenario_3 = mean_squared_error(y_test_scaled, y_pred)
        mae_scenario_3 = mean_absolute_error(y_test_scaled, y_pred)
        r2_scenario_3 = r2_score(y_test_scaled, y_pred)
        rmse_scenario_3 = np.sqrt(mse_scenario_1)
        
        # Store the metrics
        metrics_scenario_1[name] = {
            'MSE': mse_scenario_3,
            'MAE': mae_scenario_3,
            'R²': r2_scenario_3,
            'RMSE': rmse_scenario_3
        }
        
        # Display the metrics
        print(f"Metrics scenario_1 for {name}:")
        print(f"  MSE: {mse_scenario_3}")
        print(f"  MAE: {mae_scenario_3}")
        print(f"  R²: {r2_scenario_3}")
        print(f"  RMSE: {rmse_scenario_3}")
        print("\n" + "-"*50 + "\n")


    # Scenario 4 Train: Label encoding for 'Quarter' and one-hot encoding for 'PoliceStationCode'
    data_onehot_label_encoded_train = X_train_display.copy()
    data_onehot_label_encoded_train['Quarter'] = label_encoder_qtr_train.fit_transform(data_onehot_label_encoded_train['Quarter'])
    encoded_psc_train = onehot_encoder_train.fit_transform(data_onehot_label_encoded_train[['PoliceStationCode']]).toarray()
    encoded_psc_df_train = pd.DataFrame(encoded_psc_train, columns=onehot_encoder_train.get_feature_names_out(['PoliceStationCode']))
    data_onehot_label_encoded_train = pd.concat([data_onehot_label_encoded_train, encoded_psc_df_train], axis=1).drop(['PoliceStationCode'], axis=1)


    # Apply the same encoding to the test set
    data_onehot_label_encoded_test = X_test_display.copy()
    data_onehot_label_encoded_test['Quarter'] = label_encoder_qtr_train.fit_transform(data_onehot_label_encoded_test['Quarter'])
    encoded_psc_test = onehot_encoder_train.fit_transform(data_onehot_label_encoded_test[['PoliceStationCode']]).toarray()
    encoded_psc_df_test = pd.DataFrame(encoded_psc_test, columns=onehot_encoder_train.get_feature_names_out(['PoliceStationCode']))
    data_onehot_label_encoded_test = pd.concat([data_onehot_label_encoded_test, encoded_psc_df_test], axis=1).drop(['PoliceStationCode'], axis=1)

    # Select only numerical columns
    numerical_columns_train = data_onehot_label_encoded_train.select_dtypes(include=['float64', 'int64']).columns
    data_onehot_label_encoded_train = data_onehot_label_encoded_train[numerical_columns_train]
    data_onehot_label_encoded_test = data_onehot_label_encoded_test[numerical_columns_train]

    # Apply StandardScaler
    scaler = StandardScaler()
    data_onehot_label_encoded_train_scaled = scaler.fit_transform(data_onehot_label_encoded_train)
    data_onehot_label_encoded_test_scaled = scaler.transform(data_onehot_label_encoded_test)

    # Train and Evaluate
    metrics_scenario_4 = {}
    best_models_scenario_4 = {}
    predictions_scenario_4 = {}
    metrics_scenario_4 = {}

    for name, model in models.items():
        print(f"Training and predicting with {name}...")

        print(f'This is your made name : {name}')
        
        # Perform GridSearchCV to find the best parameters
        grid_search = GridSearchCV(estimator=model, param_grid=params[name], cv=5, n_jobs=-1, scoring='r2')
        grid_search.fit(X_train_scaled, y_train_scaled)
        
        # Store the best model
        best_models_scenario_1[name] = grid_search.best_estimator_
        
        # Make predictions on the test set
        y_pred = grid_search.predict(X_test_scaled)
        predictions_scenario_1[name] = y_pred
        
        # Calculate evaluation metrics
        mse_scenario_4 = mean_squared_error(y_test_scaled, y_pred)
        mae_scenario_4 = mean_absolute_error(y_test_scaled, y_pred)
        r2_scenario_4 = r2_score(y_test_scaled, y_pred)
        rmse_scenario_4 = np.sqrt(mse_scenario_1)
        
        # Store the metrics
        metrics_scenario_1[name] = {
            'MSE': mse_scenario_4,
            'MAE': mae_scenario_4,
            'R²': r2_scenario_4,
            'RMSE': rmse_scenario_4
        }
        
        # Display the metrics
        print(f"Metrics scenario_1 for {name}:")
        print(f"  MSE: {mse_scenario_4}")
        print(f"  MAE: {mae_scenario_4}")
        print(f"  R²: {r2_scenario_4}")
        print(f"  RMSE: {rmse_scenario_4}")
        print("\n" + "-"*50 + "\n")


    # ===================================================================================================
with SplitScalerTab4:
    with st.expander('Train and test split', expanded=False):
        st.header(f'Input data for {algorithm} algorithm', divider='rainbow')
        train_ratio = parameter_split_size
        test_ratio = 100 - parameter_split_size
        split_ration_value = f'{train_ratio} : {test_ratio}'
        split_ration = f'Split Ration % Train\:Test'
    
        # st.write(param_grid)
        # st.write(model)
        #st.write(X_train_display.dtypes)

        col = st.columns(5)
        col[0].metric(label="No. of samples", value=X.shape[0], delta="")
        col[1].metric(label="No. of X variables", value=X.shape[1], delta="")
        col[2].metric(label="No. of Training samples", value=X_train.shape[0], delta="")
        col[3].metric(label="No. of Test samples", value=X_test.shape[0], delta="")
        col[4].metric(label= split_ration, value= split_ration_value, delta="")

    
    # Display the updated train and test splits
    with st.expander('Train split : MinMaxScaler', expanded=False):
        train_col = st.columns((3, 1))
        with train_col[0]:
            st.markdown('**X_train**')
            st.dataframe(X_train_display, height=210, hide_index=True, use_container_width=True)
        with train_col[1]:
            st.markdown('**y_train**')
            st.dataframe(y_train_scaled, height=210, hide_index=True, use_container_width=True)

    with st.expander('Test split : MinMaxScaler', expanded=False):
        test_col = st.columns((3, 1))
        with test_col[0]:
            st.markdown('**X_test**')
            st.dataframe(X_test_display, height=210, hide_index=True, use_container_width=True)
        with test_col[1]:
            st.markdown('**y_test**')
            st.dataframe(y_test_scaled, height=210, hide_index=True, use_container_width=True)
with EncodeTab5:
    with st.expander(f'Predictions and Mertics: Interger Encoding', expanded=False):
        st.header(f'Train :', divider='rainbow')
        # st.dataframe(data_label_encoded_train, height=210, hide_index=True, use_container_width=True)
        st.header(f'Test :', divider='rainbow')
        # st.dataframe(data_label_encoded_test, height=210, hide_index=True, use_container_width=True)

    with st.expander('Label encoding Data : PoliceStationCode and Quarter', expanded=False):
        st.header(f'Label encoding Train', divider='rainbow')
        st.dataframe(data_label_encoded_train, height=210, hide_index=True, use_container_width=True)
        st.header(f'Label encoding Test', divider='rainbow')
        st.dataframe(data_label_encoded_test, height=210, hide_index=True, use_container_width=True)
  
    with st.expander('One-hot encoding Data : PoliceStationCode and Quarter', expanded=False):
        st.header(f'One-hot encoding Train', divider='rainbow')
        st.dataframe(data_onehot_encoded_train, height=210, hide_index=True, use_container_width=True)
        st.header(f'One-hot encoding Test', divider='rainbow')
        st.dataframe(data_onehot_encoded_test, height=210, hide_index=True, use_container_width=True)

    with st.expander('Label encoding Data : PoliceStationCode and One-hot encoding : Quarter', expanded=False):
        st.header(f'Label encoding Train : One-hot encoding', divider='rainbow')
        st.dataframe(data_label_onehot_encoded_train, height=210, hide_index=True, use_container_width=True)
        st.header(f'Label encoding Test : One-hot encoding', divider='rainbow')
        st.dataframe(data_label_onehot_encoded_test, height=210, hide_index=True, use_container_width=True)

    with st.expander('One-hot encoding Data : PoliceStationCode and Label encoding : Quarter', expanded=False):
        st.header(f'Label encoding Train : One-hot encoding', divider='rainbow')
        st.dataframe(data_onehot_label_encoded_train, height=210, hide_index=True, use_container_width=True)
        st.header(f'Label encoding Test : One-hot encoding', divider='rainbow')
        st.dataframe(data_onehot_label_encoded_test, height=210, hide_index=True, use_container_width=True)

with RFMPredictionTab6:
    with st.expander(f'Predictions and Mertics: Interger Encoding', expanded=False):
        st.header(f'Train :', divider='rainbow')
        # st.dataframe(data_label_encoded_train, height=210, hide_index=True, use_container_width=True)
        st.header(f'Test :', divider='rainbow')
        # st.dataframe(data_label_encoded_test, height=210, hide_index=True, use_container_width=True)
    
    with st.expander(f'Predictions and Mertics: Label encoding : PoliceStationCode and Quarter', expanded=False):
        st.header(f'Train :', divider='rainbow')
        # st.dataframe(data_label_encoded_train, height=210, hide_index=True, use_container_width=True)
        st.header(f'Test :', divider='rainbow')
        # st.dataframe(data_label_encoded_test, height=210, hide_index=True, use_container_width=True)

    with st.expander('Predictions and Mertics: One-hot encoding Predictions : PoliceStationCode and Quarter', expanded=False):
        st.header(f'Train :', divider='rainbow')
        # st.dataframe(data_onehot_encoded_train, height=210, hide_index=True, use_container_width=True)
        st.header(f'Test :', divider='rainbow')
        # st.dataframe(data_onehot_encoded_test, height=210, hide_index=True, use_container_width=True)

    with st.expander('Predictions and Mertics : Label encoding : PoliceStationCode and One-hot encoding : Quarter', expanded=False):
        st.header(f'Train :', divider='rainbow')
        # st.dataframe(data_label_onehot_encoded_train, height=210, hide_index=True, use_container_width=True)
        st.header(f'Test :', divider='rainbow')
        # st.dataframe(data_label_onehot_encoded_test, height=210, hide_index=True, use_container_width=True) 
         
    with st.expander('Predictions and Mertics : One-hot encoding: PoliceStationCode and Label encoding : Quarter', expanded=False):
        st.header(f'Train :', divider='rainbow')
        # st.dataframe(data_onehot_label_encoded_train, height=210, hide_index=True, use_container_width=True)
        st.header(f'Test :', divider='rainbow')
        # st.dataframe(data_onehot_label_encoded_test, height=210, hide_index=True, use_container_width=True)  

with SVRPredictionTab7:
    with st.expander(f'Predictions and Mertics: Interger Encoding', expanded=False):
        st.header(f'Train :', divider='rainbow')
        # st.dataframe(data_label_encoded_train, height=210, hide_index=True, use_container_width=True)
        st.header(f'Test :', divider='rainbow')
        # st.dataframe(data_label_encoded_test, height=210, hide_index=True, use_container_width=True)
    
with XGBoostPredictionTab8:
    with st.expander(f'Predictions and Mertics: Interger Encoding', expanded=False):
        st.header(f'Train :', divider='rainbow')
        # st.dataframe(data_label_encoded_train, height=210, hide_index=True, use_container_width=True)
        st.header(f'Test :', divider='rainbow')
        # st.dataframe(data_label_encoded_test, height=210, hide_index=True, use_container_width=True)
    
    with st.expander(f'Predictions and Mertics: Label encoding : PoliceStationCode and Quarter', expanded=False):
        st.header(f'Train :', divider='rainbow')
        # st.dataframe(data_label_encoded_train, height=210, hide_index=True, use_container_width=True)
        st.header(f'Test :', divider='rainbow')
        # st.dataframe(data_label_encoded_test, height=210, hide_index=True, use_container_width=True)

    with st.expander('Predictions and Mertics: One-hot encoding Predictions : PoliceStationCode and Quarter', expanded=False):
        st.header(f'Train :', divider='rainbow')
        # st.dataframe(data_onehot_encoded_train, height=210, hide_index=True, use_container_width=True)
        st.header(f'Test :', divider='rainbow')
        # st.dataframe(data_onehot_encoded_test, height=210, hide_index=True, use_container_width=True)

    with st.expander('Predictions and Mertics : Label encoding : PoliceStationCode and One-hot encoding : Quarter', expanded=False):
        st.header(f'Train :', divider='rainbow')
        # st.dataframe(data_label_onehot_encoded_train, height=210, hide_index=True, use_container_width=True)
        st.header(f'Test :', divider='rainbow')
        # st.dataframe(data_label_onehot_encoded_test, height=210, hide_index=True, use_container_width=True) 
         
    with st.expander('Predictions and Mertics : One-hot encoding: PoliceStationCode and Label encoding : Quarter', expanded=False):
        st.header(f'Train :', divider='rainbow')
        # st.dataframe(data_onehot_label_encoded_train, height=210, hide_index=True, use_container_width=True)
        st.header(f'Test :', divider='rainbow')
        # st.dataframe(data_onehot_label_encoded_test, height=210, hide_index=True, use_container_width=True)  


with ANNPredictionTab9:
    with st.expander(f'Predictions and Mertics: Interger Encoding', expanded=False):
        st.header(f'Train :', divider='rainbow')
        # st.dataframe(data_label_encoded_train, height=210, hide_index=True, use_container_width=True)
        st.header(f'Test :', divider='rainbow')
        # st.dataframe(data_label_encoded_test, height=210, hide_index=True, use_container_width=True)
    
    with st.expander(f'Predictions and Mertics: Label encoding : PoliceStationCode and Quarter', expanded=False):
        st.header(f'Train :', divider='rainbow')
        # st.dataframe(data_label_encoded_train, height=210, hide_index=True, use_container_width=True)
        st.header(f'Test :', divider='rainbow')
        # st.dataframe(data_label_encoded_test, height=210, hide_index=True, use_container_width=True)

    with st.expander('Predictions and Mertics: One-hot encoding Predictions : PoliceStationCode and Quarter', expanded=False):
        st.header(f'Train :', divider='rainbow')
        # st.dataframe(data_onehot_encoded_train, height=210, hide_index=True, use_container_width=True)
        st.header(f'Test :', divider='rainbow')
        # st.dataframe(data_onehot_encoded_test, height=210, hide_index=True, use_container_width=True)

    with st.expander('Predictions and Mertics : Label encoding : PoliceStationCode and One-hot encoding : Quarter', expanded=False):
        st.header(f'Train :', divider='rainbow')
        # st.dataframe(data_label_onehot_encoded_train, height=210, hide_index=True, use_container_width=True)
        st.header(f'Test :', divider='rainbow')
        # st.dataframe(data_label_onehot_encoded_test, height=210, hide_index=True, use_container_width=True) 
         
    with st.expander('Predictions and Mertics : One-hot encoding: PoliceStationCode and Label encoding : Quarter', expanded=False):
        st.header(f'Train :', divider='rainbow')
        # st.dataframe(data_onehot_label_encoded_train, height=210, hide_index=True, use_container_width=True)
        st.header(f'Test :', divider='rainbow')
        # st.dataframe(data_onehot_label_encoded_test, height=210, hide_index=True, use_container_width=True)  

with KNNPredictionTab10:
    with st.expander(f'Predictions and Mertics: Interger Encoding', expanded=False):
        st.header(f'Train :', divider='rainbow')
        # st.dataframe(data_label_encoded_train, height=210, hide_index=True, use_container_width=True)
        st.header(f'Test :', divider='rainbow')
        # st.dataframe(data_label_encoded_test, height=210, hide_index=True, use_container_width=True)
    
    with st.expander(f'Predictions and Mertics: Label encoding : PoliceStationCode and Quarter', expanded=False):
        st.header(f'Train :', divider='rainbow')
        # st.dataframe(data_label_encoded_train, height=210, hide_index=True, use_container_width=True)
        st.header(f'Test :', divider='rainbow')
        # st.dataframe(data_label_encoded_test, height=210, hide_index=True, use_container_width=True)

    with st.expander('Predictions and Mertics: One-hot encoding Predictions : PoliceStationCode and Quarter', expanded=False):
        st.header(f'Train :', divider='rainbow')
        # st.dataframe(data_onehot_encoded_train, height=210, hide_index=True, use_container_width=True)
        st.header(f'Test :', divider='rainbow')
        # st.dataframe(data_onehot_encoded_test, height=210, hide_index=True, use_container_width=True)

    with st.expander('Predictions and Mertics : Label encoding : PoliceStationCode and One-hot encoding : Quarter', expanded=False):
        st.header(f'Train :', divider='rainbow')
        # st.dataframe(data_label_onehot_encoded_train, height=210, hide_index=True, use_container_width=True)
        st.header(f'Test :', divider='rainbow')
        # st.dataframe(data_label_onehot_encoded_test, height=210, hide_index=True, use_container_width=True) 
         
    with st.expander('Predictions and Mertics : One-hot encoding: PoliceStationCode and Label encoding : Quarter', expanded=False):
        st.header(f'Train :', divider='rainbow')
        # st.dataframe(data_onehot_label_encoded_train, height=210, hide_index=True, use_container_width=True)
        st.header(f'Test :', divider='rainbow')
        # st.dataframe(data_onehot_label_encoded_test, height=210, hide_index=True, use_container_width=True)  

with AdHocTab11:
    with st.expander(f'AdHoc- Analysis : Interger Encoding', expanded=False):
        st.header(f'RFM - Train :', divider='rainbow')
        # st.dataframe(data_label_encoded_train, height=210, hide_index=True, use_container_width=True)
        st.header(f'RFM - Test :', divider='rainbow')
        # st.dataframe(data_label_encoded_test, height=210, hide_index=True, use_container_width=True)
    
    with st.expander(f'AdHoc- Analysis : Label encoding : PoliceStationCode and Quarter', expanded=False):
        st.header(f'Train :', divider='rainbow')
        # st.dataframe(data_label_encoded_train, height=210, hide_index=True, use_container_width=True)
        st.header(f'Test :', divider='rainbow')
        # st.dataframe(data_label_encoded_test, height=210, hide_index=True, use_container_width=True)

    with st.expander('AdHoc- Analysis : One-hot encoding Predictions : PoliceStationCode and Quarter', expanded=False):
        st.header(f'Train :', divider='rainbow')
        # st.dataframe(data_onehot_encoded_train, height=210, hide_index=True, use_container_width=True)
        st.header(f'Test :', divider='rainbow')
        # st.dataframe(data_onehot_encoded_test, height=210, hide_index=True, use_container_width=True)

    with st.expander('AdHoc- Analysis : Label encoding : PoliceStationCode and One-hot encoding : Quarter', expanded=False):
        st.header(f'Train :', divider='rainbow')
        # st.dataframe(data_label_onehot_encoded_train, height=210, hide_index=True, use_container_width=True)
        st.header(f'Test :', divider='rainbow')
        # st.dataframe(data_label_onehot_encoded_test, height=210, hide_index=True, use_container_width=True) 
         
    with st.expander('AdHoc- Analysis : One-hot encoding: PoliceStationCode and Label encoding : Quarter', expanded=False):
        st.header(f'Train :', divider='rainbow')
        # st.dataframe(data_onehot_label_encoded_train, height=210, hide_index=True, use_container_width=True)
        st.header(f'Test :', divider='rainbow')
        # st.dataframe(data_onehot_label_encoded_test, height=210, hide_index=True, use_container_width=True)  