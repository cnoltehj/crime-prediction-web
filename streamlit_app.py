import streamlit as st
import pandas as pd
import numpy as np
import requests
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
import matplotlib.cm as cm
import seaborn as sns
import shap
from dataRequest.crimedbRequest import (
    fetch_all_provinces,
    fetch_policestation_per_provinces,
    fetch_stats_province_policestation,
    fetch_predition_province_policestation_year_quarterly_algorithm,
    fetch_suggest_stats_province_policestation,
    fetch_stats_policestation_per_province,
    fetch_training_metrics_data,
    save_train_prediction_data,
    save_all_prediction_data,
    save_metric_data
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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler , StandardScaler
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

st.title('Interpretable Crime Hotspot Prediction')

AboutTab1,PreditionsUserViewTab2,PostHocAnalysisTab3,DataExplorationTab4,Transformationtab5,SplitScalerDataTab6, EncodingDataTab7,ModelTrainingTab8,DBTrainedValuesTab9,DBAllPreditionsTab10 = st.tabs(['About','Preditions-User','PostHocAnalysis','Exploration-Data','Transformation-Data','ScalerSplit-Data', 'Encoded-Data','ModelTraining','DBTrainedValues','AllPreditions'])

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
            #set_development_mode = st.toggle('Switch Development Mode') # Switch to Auto Inject Development Mode

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

                    police_station_name = st.selectbox('Select Police Station', df_policestations['StationName'], format_func=lambda x: x, index=2)
                    police_code_value = df_policestations[df_policestations['StationName'] == police_station_name]['StationCode'].values[0]
                    year_mapping = st.slider('Select year range from 2016 - 2023', 2023, 2016)
                    quarter_value = st.radio('Select quarter of year', options=[1, 2, 3, 4], index=0)
                    df_province_policestation_quarterly_data_db = fetch_stats_province_policestation(province_code_value, police_code_value, quarter_value)
                    df_suggeted_province_quarterly_data_db = fetch_stats_policestation_per_province(province_code_value,quarter_value)
                    df_suggeted_province_policestation_quarterly_data_db = fetch_suggest_stats_province_policestation(province_code_value,police_code_value)
   
            st.subheader('2. Select Algorithm')
            with st.expander('Algorithms'):
                algorithm = st.radio('', options=['ANN (MLPRegressor)', 'KNN', 'RFM', 'SVR','XGBoost'], index=4)
                
            st.markdown('**1.4. Set Test and Train Parameters**')
            parameter_split_size = st.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

            st.subheader('3. Learning Parameters')
            with st.expander('See parameters', expanded=False):
                sleep_time = st.slider('Sleep time', 0, 3, 0)

        if not df_crime_data_db.empty:
            with st.status("Running ...", expanded=True) as status:

                # st.write("Loading data ...")
                # time.sleep(sleep_time)

                # st.write("Preparing data ...")
                # time.sleep(sleep_time)

                # Initialize empty lists for metrics and crime categories
                crime_categories_list = df_crime_data_db['CrimeCategory'].tolist()

with PreditionsUserViewTab2:
        # Fetch the prediction data
        def_fetch_stats_province_policestation_quarterly_algorithm_db = fetch_predition_province_policestation_year_quarterly_algorithm(province_code_value, police_code_value, quarter_value, "MLPR")


        st.header(f'{province_name}: Predictions vs True_Value', divider='rainbow')
        df_prediction_ui = pd.DataFrame(def_fetch_stats_province_policestation_quarterly_algorithm_db)
                
        # Bar plot for Prediction vs True_Value
        bar_width = 0.35
        index = range(len(df_prediction_ui))

                # Create the bar plot
        fig, ax = plt.subplots()
        bar1 = ax.bar(index, df_prediction_ui['Prediction'], bar_width, label='Prediction', color='b')
        bar2 = ax.bar([i + bar_width for i in index], df_prediction_ui['True_Value'], bar_width, label='True Value', color='r')

        ax.set_xlabel('CrimeCategory')
        ax.set_ylabel('Counts')
        ax.set_title(f'{province_code_value}: {police_station_name} : Prediction vs True Value')
        ax.set_xticks([i + bar_width / 2 for i in index])
        ax.set_xticklabels(df_prediction_ui['CrimeCategory'], rotation=45, ha='right')
        ax.legend()

                # Display the bar plot in Streamlit
        st.pyplot(fig)

                #st.header('Predictions plot', divider='rainbow')
                # Header for the first section
        st.header(f'{province_code_value}: {police_station_name} Police Station : Predictions', divider='rainbow')
        st.dataframe(def_fetch_stats_province_policestation_quarterly_algorithm_db) #.sort_values(by='CrimeCategory'))
    
    

with DataExplorationTab4:
    st.header(f'{province_name}: {police_station_name} Police station Initial dataset', divider='rainbow')
    if not def_fetch_stats_province_policestation_quarterly_algorithm_db.empty:
        st.dataframe(def_fetch_stats_province_policestation_quarterly_algorithm_db)  # Show the dataframe
    else:
        st.write("No data available or API returned an empty result.")

    st.header(f'{province_name}: All initial dataset used for training models', divider='rainbow')
    st.dataframe(df_suggeted_province_quarterly_data_db) #.sort_values(by='PoliceStationCode'))

with Transformationtab5:
        st.header('Identify outliers', divider='rainbow')
        performance_col = st.columns((2, 0.2, 3))

        with performance_col[0]:
            st.header('Outliers', divider='rainbow')
            df_identify_outliers_db = identify_outliers_data(df_suggeted_province_quarterly_data_db)
            st.dataframe(df_identify_outliers_db.sort_values(by='PoliceStationCode'))

                # Plot box plot of the data with outliers replaced
        with performance_col[2]:
            st.header('Outliers percentage plot', divider='rainbow')
            # Melt DataFrame to long format for easy plotting
              # if identify_outlier:
            df_identify_outliers_db = identify_outliers_data(df_suggeted_province_quarterly_data_db)
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
        st.header('Replaced outliers with the median value', divider='rainbow')
        df_replace_outliers = replace_outliers_data(df_suggeted_province_quarterly_data_db)
        st.dataframe(df_replace_outliers.sort_values(by='PoliceStationCode'), height=210, use_container_width=True)

# Initialize models and parameter grid
models = {
    'RFM': RandomForestRegressor(),
    'SVR': SVR(),
    'XGBR': XGBRegressor(),
    'KNNR': KNeighborsRegressor(),
    'MLPR': MLPRegressor(max_iter=1000)
 }

params = {
    'RFM': {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]},
    'SVR': {'C': [1, 10], 'kernel': ['linear', 'rbf']},
    'XGBR': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]},
    'KNNR': {'n_neighbors': [5, 10], 'weights': ['uniform', 'distance']},
    'MLPR': {'hidden_layer_sizes': [(100,), (100, 50)], 'activation': ['relu', 'tanh'],'learning_rate_init': [0.001, 0.01],'max_iter': [500, 1000],'solver': ['adam', 'lbfgs']}
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

df_wide = replace_non_finite_with_median(df_cleaned)

# B - Defining feature set X (years 2016-2023) and target y (2024 prediction) =====
# We exclude 'CrimeCategory', 'ProvinceCode', 'PoliceStationCode', 'Quarter' as these are categorical
features = [col for col in df_wide.columns if col not in ['CrimeCategory', 'ProvinceCode', 'Quarter', '2023']]

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

# 2. Standardize/Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Function to replace non-finite and NaN values with the median
def replace_non_finite_with_median(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.apply(lambda x: x.fillna(x.median()) if np.issubdtype(x.dtype, np.number) else x)
    return df

# Function to ensure consistent data types across DataFrame
def ensure_consistent_dtypes(df):
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype('float64')
        else:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].astype('float64')
            except ValueError:
                pass
    return df

# Function to train and predict per scenario and algorithm
def train_and_predict(df, models, params, scenario_func, scenario_name, original_df):
    df_encoded = scenario_func(df.copy())
    
    features = [col for col in df_encoded.columns if col not in ['CrimeCategory', 'ProvinceCode', 'PoliceStationCode', 'Quarter', '2023']]
    
    realtime_predictions = []
    realtime_metrics = []

    for name, model in models.items():
        grid_search = GridSearchCV(model, params[name], cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
        grid_search.fit(X_train_scaled, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test_scaled)
        
        # Adjust the length of the predictions DataFrame to match the test set
        scenario_predictions = pd.DataFrame({
            'CrimeCategory': original_df.loc[y_test.index, 'CrimeCategory'].values,
            'ProvinceCode': original_df.loc[y_test.index, 'ProvinceCode'].values,
            'PoliceStationCode': original_df.loc[y_test.index, 'PoliceStationCode'].values,
            'Quarter': original_df.loc[y_test.index, 'Quarter'].values,
            'Algorithm': [name] * len(y_test),
            'Scenario': [scenario_name] * len(y_test),
            'Prediction': y_pred,
            'True_value': y_test
        })

        # Store metrics with algorithm and scenario name
        scenario_metrics = pd.DataFrame({
            'Algorithm': [name],
            'Scenario': [scenario_name],
            'MAE': [mean_absolute_error(y_test, y_pred)],
            'MSE': [mean_squared_error(y_test, y_pred)],
            'R²': [r2_score(y_test, y_pred)],
            'MAPE': [mean_absolute_percentage_error(y_test, y_pred)],
            'ARS': [adjusted_rand_score(y_test, y_pred)]
        })

        # Append the results to the respective lists
        realtime_predictions.append(scenario_predictions)
        realtime_metrics.append(scenario_metrics)

    # Concatenate all predictions and metrics after the loop ends
    predictions_df = pd.concat(realtime_predictions, axis=0).reset_index(drop=True)
    metrics_df = pd.concat(realtime_metrics, axis=0).reset_index(drop=True)
    
    # Save all the predictions and metrics to the database
    if not predictions_df.empty:
        save_train_prediction_data(predictions_df)
    
    if not metrics_df.empty:
        save_metric_data(metrics_df)

    return predictions_df, metrics_df


# Scenario functions
def scenario_0(df):
    # In this scenario, no encoding is applied. The raw data as it is.
    return df

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

def scenario_5(df):
    # Integer encode categorical variables using LabelEncoder for 'PoliceStationCode' and 'Quarter'
    label_encoder_psc = LabelEncoder()
    label_encoder_qtr = LabelEncoder()
    
    df['PoliceStationCode'] = label_encoder_psc.fit_transform(df['PoliceStationCode'])
    df['Quarter'] = label_encoder_qtr.fit_transform(df['Quarter'])
    
    return df


# Replace non-finite and NaN values with median
df_cleaned = replace_non_finite_with_median(df_replace_outliers)


# Prepare train/test data
X = df_cleaned[['2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']]
y = df_cleaned['2023']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Run scenarios
scenario_funcs = {

    "Scenario 0": scenario_0,  # No encoding
    "Scenario 1": scenario_1,  # Label encoding for 'PoliceStationCode' and 'Quarter'
    "Scenario 2": scenario_2,  # One-hot encoding for 'PoliceStationCode' and 'Quarter'
    "Scenario 3": scenario_3,  # Label encoding for 'PoliceStationCode' and One-hot for 'Quarter'
    "Scenario 4": scenario_4,  # Label encoding for 'Quarter' and One-hot for 'PoliceStationCode'
    "Scenario 5": scenario_5   # Integer encoding
}
          
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

with SplitScalerDataTab6:
     # Display data info
    st.header('Input data', divider='rainbow')
    col = st.columns(4)
    col[0].metric(label="No. of samples", value=X.shape[0], delta="")
    col[1].metric(label="No. of X variables", value=X.shape[1], delta="")
    col[2].metric(label="No. of Training samples", value=X_train.shape[0], delta="")
    col[3].metric(label="No. of Test samples", value=X_test.shape[0], delta="")
    
    with st.expander(f'Trained scaled', expanded=False):
        train_col = st.columns((3,1))
        with train_col[0]:
            st.markdown('**X**')
            st.dataframe(X_train, height=210, hide_index=True, use_container_width=True)
        with train_col[1]:
            st.markdown('**y**')
            st.dataframe(y_train, height=210, hide_index=True, use_container_width=True)
    with st.expander('Test split', expanded=False):
        test_col = st.columns((3,1))
        with test_col[0]:
            st.markdown('**X**')
            st.dataframe(X_test, height=210, hide_index=True, use_container_width=True)
        with test_col[1]:
            st.markdown('**y**')
            st.dataframe(y_test, height=210, hide_index=True, use_container_width=True)

with ModelTrainingTab8:
    trainmodel = st.toggle('Train model')
    preditionsmodel = st.toggle('Prediction after train the model')

    if trainmodel:
        for scenario_name, scenario_func in scenario_funcs.items():
            for name in models.keys():
                # Train and predict for each scenario and model
                scenario_predictions, scenario_metrics = train_and_predict(
                    df_cleaned, {name: models[name]}, {name: params[name]}, scenario_func, scenario_name, df_cleaned
                )

                # Display predictions and metrics for each algorithm and scenario
                st.markdown(f"Predictions for {name} under {scenario_name}")
                st.dataframe(scenario_predictions, height=300, use_container_width=True)
                
                st.markdown(f"Metrics for {name} under {scenario_name}")
                st.dataframe(scenario_metrics.head(1), height=50, use_container_width=True)

    if preditionsmodel:
       
        st.markdown(f"Predictions")
        # st.dataframe(scenario_predictions, height=300, use_container_width=True)


with DBTrainedValuesTab9:

  # Run scenarios
    metrics_scenario = {
        "Scenario 0": 0,  # No encoding
        "Scenario 1": 1,  # Label encoding for 'PoliceStationCode' and 'Quarter'
        "Scenario 2": 2,  # One-hot encoding for 'PoliceStationCode' and 'Quarter'
        "Scenario 3": 3,  # Label encoding for 'PoliceStationCode' and One-hot for 'Quarter'
        "Scenario 4": 4,  # Label encoding for 'Quarter' and One-hot for 'PoliceStationCode'
        "Scenario 5": 5   # Integer encoding
        }

        # Fetch the training metrics data for all scenarios
    df_training_metrics_data_db = fetch_training_metrics_data()

    with st.expander(f'Metrics Bar Plot', expanded=False):
        st.header(f'Train Metrics plots :', divider='rainbow')

        # Metrics to be plotted
        metrics = ['MAE', 'MSE', 'R-Square', 'MAPE', 'ARS']
        models = df_training_metrics_data_db['Algorithm'].unique()

        # Initialize scenario names for the x-axis labels
        scenario_labels = list(metrics_scenario.keys())

        # Define a colormap to give each bar a different color per model
        colors = cm.get_cmap('tab10', len(models))

        # Plot the bar charts for each metric across all models and scenarios
        for metric in metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Width for each bar group
            bar_width = 0.15
            
            # X-axis indices for each scenario
            indices = np.arange(len(scenario_labels))
            
            # Loop through each model and plot bars for each scenario
            for i, model in enumerate(models):
                model_data = []
                
                for scenario_name, scenario_id in metrics_scenario.items():
                    # Extract the row for the specific scenario and model
                    scenario_metrics = df_training_metrics_data_db[
                        (df_training_metrics_data_db['Scenario'] == scenario_id) &
                        (df_training_metrics_data_db['Algorithm'] == model)
                    ]
                    if not scenario_metrics.empty:
                        model_data.append(scenario_metrics[metric].values[0])
                    else:
                        model_data.append(np.nan)
                
                # Plot the bar chart for the current model and metric
                bars = ax.bar(indices + i * bar_width, model_data, bar_width, label=model, color=colors(i))
                
                # Add values on top of each bar
                for bar in bars:
                    yval = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

            # Set chart labels and title
            ax.set_xticks(indices + bar_width * (len(models) - 1) / 2)
            ax.set_xticklabels(scenario_labels, rotation=45, ha='right')
            ax.set_ylabel(f'{metric} Value')
            ax.set_title(f'{metric} for Each Model across Scenarios')

            # Add a legend to the graph
            ax.legend(title='Models', loc='best')

            # Display the chart in Streamlit
            st.pyplot(fig)

    with st.expander(f'Get Metrics Data', expanded=False):
        st.header(f'Get Train Metrics data from database :', divider='rainbow')

        # Metrics to be plotted
        metrics = ['MAE', 'MSE', 'R-Square', 'MAPE', 'ARS'] #'R²'

        # Initialize a dictionary to hold values for each metric across all scenarios
        metrics_values = {metric: [] for metric in metrics}

        # Initialize scenario names for the x-axis labels
        scenario_labels = list(metrics_scenario.keys())

        # Populate the dictionary with metric values for each scenario
        for scenario_name, scenario_id in metrics_scenario.items():
            # Extract the rows from the dataframe for the specific scenario
            scenario_metrics = df_training_metrics_data_db[df_training_metrics_data_db['Scenario'] == scenario_id]
            
            st.subheader(f'{scenario_name}', divider='rainbow')
            
            # Filter the dataframe to get metrics for the current scenario
            scenario_metrics = df_training_metrics_data_db[df_training_metrics_data_db['Scenario'] == scenario_id]
            
            # Display the metrics for the current scenario
            st.dataframe(scenario_metrics, height=210, hide_index=True, use_container_width=True)
         

    with st.expander(f'Get Trained-Preditions from database', expanded=False):
        st.header(f'Train :', divider='rainbow')
        # st.dataframe(final_predictions, height=210, hide_index=True, use_container_width=True)
        # st.header(f'Test :', divider='rainbow')
        # st.dataframe(final_metrics, height=210, hide_index=True, use_container_width=True)

with DBAllPreditionsTab10:
    with st.expander(f'All best model prediction extracted from the database', expanded=False):
        st.header(f'Scaled :', divider='rainbow')
        st.dataframe(X_train_scaled, height=210, hide_index=True, use_container_width=True)

    with st.expander(f'All best model prediction generated after training', expanded=False):
        st.header(f'Prediction from best model MLRP Scenario 5 :', divider='rainbow')

        # Initialize models and parameter grid
        models_predict = {
            'MLPR': MLPRegressor(max_iter=1000)
        }

        params_predict = {
            'MLPR': {'hidden_layer_sizes': [(100,), (100, 50)], 'activation': ['relu', 'tanh'],
                    'learning_rate_init': [0.001, 0.01], 'max_iter': [500, 1000], 'solver': ['adam', 'lbfgs']}
        }

        def scenario_all_prediction(df):
            # Initialize label encoders
            label_encoder_psc = LabelEncoder()
            label_encoder_qtr = LabelEncoder()
            
            # Encode 'PoliceStationCode' and 'Quarter'
            df['PoliceStationCode'] = label_encoder_psc.fit_transform(df['PoliceStationCode'])
            df['Quarter'] = label_encoder_qtr.fit_transform(df['Quarter'])
            
            # Return both the encoded DataFrame and the label encoders for decoding later
            label_encoders = {
                'PoliceStationCode': label_encoder_psc,
                'Quarter': label_encoder_qtr
            }
            
            return df, label_encoders

        def predict_best_trained_scenario(df_cleaned, model_name='MLPR'):
            df_cleaned = replace_non_finite_with_median(df_cleaned)

            # Integer Encoding (Scenario 5) - Returns two values now
            df_encoded, label_encoders = scenario_all_prediction(df_cleaned)

            # Define the feature set (years 2016-2023) and target (2023 for training, predicting 2024)
            X = df_encoded[['2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']]
            y = df_encoded['2023']  # Use 2023 data for training to predict 2024

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Select the model and parameters (MLPR is chosen as the best)
            selected_model = models_predict[model_name]  # Accessing the model using model_name from models_predict dictionary
            param_grid = params_predict[model_name]  # Accessing the parameter grid

            # Hyperparameter tuning with GridSearchCV
            grid_search = GridSearchCV(selected_model, param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
            grid_search.fit(X_scaled, y)

            # Best model
            best_model = grid_search.best_estimator_

            # Generate predictions for the entire dataset
            y_pred = best_model.predict(X_scaled)

            # Prepare the final predictions DataFrame
            predictions_df = pd.DataFrame({
                'CrimeCategory': df_encoded['CrimeCategory'],
                'ProvinceCode': df_encoded['ProvinceCode'],
                'PoliceStationCode': df_encoded['PoliceStationCode'],
                'Quarter': df_encoded['Quarter'],
                'Algorithm': [model_name] * len(y_pred),
                'Scenario': [scenario_name] * len(y_pred),
                'Prediction': y_pred,
                'True_value': y
            })

            # Reverse integer encoding for clarity
            for col in label_encoders:
                predictions_df[col] = label_encoders[col].inverse_transform(df_encoded[col])

             # Save the predictions to the database
            save_all_prediction_data(predictions_df)

            return predictions_df

        # Call the function and display the results in Streamlit
        predictions_best_model_df = predict_best_trained_scenario(df_cleaned, model_name='MLPR')
        st.dataframe(predictions_best_model_df, height=210, hide_index=True, use_container_width=True)


with PostHocAnalysisTab3:
    with st.expander('Shapley Post-Hoc Analysis', expanded=False):
        performance_col = st.columns((2, 0.2, 3))

        # Types of Shaplye-Plots: 

        #  1. Summary Plot: Shows the overall feature importance by aggregating Shapley values for all predictions. 
        #     It helps you understand which features are the most impactful.
        # ============================================
        # shap.summary_plot(shap_values, X_full)
        # ============================================

        # 2. Force Plot: Explains how each feature affects the prediction for a single instance. 
        #    You can generate force plots for specific predictions.
        # ============================================
        # shap.force_plot(explainer.expected_value, shap_values[instance_idx].values, X_full.iloc[instance_idx], matplotlib=True)
        # ============================================

        # 3. Dependence Plot: Shows how a particular feature affects predictions and its interaction with another feature.
        # ============================================
        # shap.dependence_plot("some_feature", shap_values, X_full)
        # ============================================




        # model_trained = XGBRegressor(n_estimators=100, learning_rate=0.01, max_depth=3)
        #  #Prepare the features and the target variable
        # X = df_prediction_ui.drop(columns=['Id', 'Prediction', 'True_Value'])  # Drop non-feature columns
        # y = df_prediction_ui['Prediction']  # Target variable

        # # Assuming 'your_model' is the trained model you are using
        # # model = your_model  # Your pre-trained model

        # # Create a SHAP explainer
        # explainer = shap.Explainer(model_trained, X)  # Replace 'model' with your trained model

        # # Calculate SHAP values for the entire dataset
        # shap_values = explainer(X)

        # # Get unique CrimeTypeNames
        # crime_types = df_prediction_ui['CrimeTypeName'].unique()

        # # Create a plot for each CrimeTypeName
        # for crime_type in crime_types:
        #     st.header(f'SHAP Force Plot for {crime_type}', divider='rainbow')
            
        #     # Filter the DataFrame for the current CrimeTypeName
        #     crime_data = df_prediction_ui[df_prediction_ui['CrimeTypeName'] == crime_type]
            
        #     # Calculate SHAP values for the filtered data
        #     crime_X = crime_data.drop(columns=['Id', 'Prediction', 'True_Value'])
        #     crime_shap_values = explainer(crime_X)
            
        #     # Plot the force plot for the first instance of the filtered data
        #     shap.initjs()
            
        #     # Display the force plot for the first instance (or any instance you want)
        #     st.write(f"Force plot for instance index 0 of {crime_type}:")
        #     force_plot = shap.force_plot(explainer.expected_value, crime_shap_values[0], crime_X.iloc[0], matplotlib=True)
            
        #     # Show the force plot in Streamlit
        #     st.pyplot(plt.gcf())
            
        #     # Clear the plot to avoid overlap in Streamlit
        #     plt.clf()


