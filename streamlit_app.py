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
    fetch_predition_province_policestation_year_quarterly_algorithm,
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

with DataExplorationTab2:
    with st.expander(f'Initial dataset', expanded=False):
        visualise_initialdate = st.toggle('Visualise initial dataset')
        st.dataframe(df_suggeted_province_quarterly_data_db.sort_values(by='PoliceStationCode'))

        # Melting the DataFrame to get a tidy format
        df_initial_data_melt = df_suggeted_province_quarterly_data_db.melt(
            id_vars=['CrimeCategory', 'ProvinceCode', 'PoliceStationCode','Quarter'],
            var_name='Year',
            value_name='Percentage'
        )

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
        with st.expander(f'Prediction Values', expanded=False):
            performance_col = st.columns((2, 0.2, 3))

            with performance_col[0]:
                st.header('Predictions', divider='rainbow')
                
                # Fetch data from the API
                df_fetch_predition_province_policestation_year_quarterly_algorithm_data_db = fetch_predition_province_policestation_year_quarterly_algorithm(province_code_value, police_code_value, quarter_value, algorithm)

                # Check if the API response is valid
                if df_fetch_predition_province_policestation_year_quarterly_algorithm_data_db is not None:
                # Create DataFrame from the API response
                    df_prediction_ui = pd.DataFrame(df_fetch_predition_province_policestation_year_quarterly_algorithm_data_db)
    
                # Check if DataFrame is empty
                if not df_prediction_ui.empty:
                # Display the DataFrame, ensuring it shows all columns
                    st.dataframe(df_prediction_ui, height=280, use_container_width=True)
                else:
                    st.write("API call returned None. Please check the API.")

                # Plot box plot of the data with outliers replaced
            with performance_col[2]:
                #st.header('Predictions plot', divider='rainbow')
                # Header for the first section
                st.header('Predictions vs True_Value Bar Plot', divider='rainbow')

                # Bar plot for Prediction vs True_Value
                bar_width = 0.35
                index = range(len(df_prediction_ui))

                # Create the bar plot
                fig, ax = plt.subplots()
                bar1 = ax.bar(index, df_prediction_ui['Prediction'], bar_width, label='Prediction', color='b')
                bar2 = ax.bar([i + bar_width for i in index], df_prediction_ui['True_Value'], bar_width, label='True Value', color='r')

                ax.set_xlabel('Crime Types')
                ax.set_ylabel('Counts')
                ax.set_title('Prediction vs True Value')
                ax.set_xticks([i + bar_width / 2 for i in index])
                ax.set_xticklabels(df_prediction_ui['CrimeTypeName'], rotation=45, ha='right')
                ax.legend()

                # Display the bar plot in Streamlit
                st.pyplot(fig)

                st.markdown("---")  # Creates a horizontal line for separation

                st.header('Predictions vs True_Value for CrimeTypeName Line Plot', divider='rainbow')

                # Create the line plot
                fig, ax = plt.subplots()
                ax.plot(df_prediction_ui['CrimeTypeName'], df_prediction_ui['Prediction'], marker='o', label='Prediction', color='b')
                ax.plot(df_prediction_ui['CrimeTypeName'], df_prediction_ui['True_Value'], marker='o', label='True Value', color='r')

                ax.set_xlabel('Crime Type Name')
                ax.set_ylabel('Counts')
                ax.set_title('Prediction vs True Value by Crime Type')
                ax.set_xticklabels(df_prediction_ui['CrimeTypeName'], rotation=45, ha='right')
                ax.legend()

                # Display the line plot in Streamlit
                st.pyplot(fig)


    # Add your content for the second column here, e.g., another plot
    # Example: st.line_chart(data2)
                

                # # Create the graph using seaborn
                # plt.figure(figsize=(10, 6))
                # sns.lineplot(data=df_predtions_melt, x='Year', y='Percentage', hue='CrimeCategory', marker='o')
                # plt.title('Crime Trends Over the Years')
                # plt.legend(title='Crime Category', bbox_to_anchor=(1.05, 1), loc='upper left')
                # plt.xticks(rotation=45)

                #         # Annotate each data point with its value
                # for i in range(df_predtions_melt.shape[0]):
                #     plt.text(df_predtions_melt['Year'].iloc[i], df_predtions_melt['Percentage'].iloc[i],
                #     f"{df_predtions_melt['Percentage'].iloc[i]:.2f}", color='black', ha='right', va='bottom')

                # # Display the graph in Streamlit
                # st.pyplot(plt)

with MetricsTab4:
    with st.expander('Mertics', expanded=False):
        performance_col = st.columns((2, 0.2, 3))
       
with ShapleyAnalysisTab5:
    with st.expander('Shapley Post-Hoc Analysis', expanded=False):
        performance_col = st.columns((2, 0.2, 3))

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

with Transformationtab6:
    # if identify_outlier:
    with st.expander('Identify outliers', expanded=False):
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

# 1. Increase max_iter
max_iter_value = 1000  # Increase this value if necessary

# Initialize models and parameter grid
models = {
    'RFM': RandomForestRegressor(),
    'SVR': SVR(),
    'XGBR': XGBRegressor(),
    'KNNR': KNeighborsRegressor(),
    'MLPR': MLPRegressor(max_iter=1000)
 }

params = {
    'RFM': {'n_estimators': [100], 'max_depth': [10,]},
    'SVR': {'C': [1, 10], 'kernel': ['linear']},  
    'XGBR': {'n_estimators': [100], 'learning_rate': [0.01], 'max_depth': [3]},  
    'KNNR': {'n_neighbors': [5], 'weights': ['uniform']},  
    'MLPR': MLPRegressor(hidden_layer_sizes=(100,),  # Simplified to one layer with 100 neurons
                         activation='relu',
                         solver='adam',  # Use Adam solver instead of lbfgs
                         learning_rate_init=0.001,
                         max_iter=max_iter_value)
}

    # 'RFM': {'n_estimators': [100], 'max_depth': [10,]},


 # 'MLPR': MLPRegressor(hidden_layer_sizes=(100,),  # Simplified to one layer with 100 neurons
    #                      activation='relu',
    #                      solver='adam',  # Use Adam solver instead of lbfgs
    #                      learning_rate_init=0.001,
    #                      max_iter=max_iter_value,
    #                      random_state=42)

      # Reduced options
    # 'SVR': {'C': [1, 10], 'kernel': ['linear']},  # Reduced kernel options
    # 'XGBR': {'n_estimators': [100], 'learning_rate': [0.01], 'max_depth': [3]},  # Simplified
    # 'KNNR': {'n_neighbors': [5], 'weights': ['uniform']},  # Simplified
    # 'MLPR': MLPRegressor(hidden_layer_sizes=(100,),  # Simplified to one layer with 100 neurons
    #                      activation='relu',
    #                      solver='adam',  # Use Adam solver instead of lbfgs
    #                      learning_rate_init=0.001,
    #                      max_iter=max_iter_value,
    #                      random_state=42)
    # }

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
features = [col for col in df_wide.columns if col not in ['CrimeCategory', 'ProvinceCode', 'Quarter', '2023']]

X = df_wide[['2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']]
y = df_wide['2023']  # Using 2023 as a proxy for training, predicting 2024

   # # Adjust the feature set to include the encoded columns as necessary
    # features = [col for col in df_encoded.columns if col not in ['CrimeCategory', 'ProvinceCode', 'Quarter', '2023']]
    
    # X = df_encoded[features]
    # y = df_encoded['2023']

# C - Splitting the dataset into train/test sets (use 2016-2023 data for training) =====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensure X_train and X_test are DataFrames before scaling
X_train_display = pd.DataFrame(X_train, columns=X.columns)
X_train_display = pd.DataFrame(X_test, columns=X.columns)

# D - Scaling the numeric features ===== Use values for Predicting withou Encoding
scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# # 2. Standardize/Scale the data
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

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

# # Function to revert 'PoliceStationCode', 'Quarter' and 'ProvinceCode' to their original values
# def revert_labels(df, original_df):
#     df['PoliceStationCode'] = original_df['Original_PoliceStationCode']
#     df['Quarter'] = original_df['Original_Quarter']
#     df['ProvinceCode'] = original_df['Original_ProvinceCode']
#     return df

# Function to train and predict per scenario and algorithm
def train_and_predict(df, models, params, scenario_func, scenario_name, original_df):
    df_encoded = scenario_func(df.copy())
    
    features = [col for col in df_encoded.columns if col not in ['CrimeCategory', 'ProvinceCode', 'PoliceStationCode', 'Quarter', '2023']]
    
    predictions = []
    metrics = []

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

        # Revert 'PoliceStationCode' and 'Quarter' to original values
        #scenario_predictions = revert_labels(scenario_predictions, original_df.loc[y_test.index])

        # Store metrics with algorithm and scenario name
        scenario_metrics = pd.DataFrame({
            'Algorithm': [name],
            'Scenario': [scenario_name] ,
            'MAE': [mean_absolute_error(y_test, y_pred)],
            'MSE': [mean_squared_error(y_test, y_pred)],
            'R²': [r2_score(y_test, y_pred)],
            'MAPE': [mean_absolute_percentage_error(y_test, y_pred)],
            'ARS': [adjusted_rand_score(y_test, y_pred)]
        })

        predictions.append(scenario_predictions)
        metrics.append(scenario_metrics)

    predictions_df = pd.concat(predictions, axis=0).reset_index(drop=True)
    metrics_df = pd.concat(metrics, axis=0).reset_index(drop=True)

    return predictions_df, metrics_df

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

# Initialize models and parameter grid
models = {
    'RFM': RandomForestRegressor(),
    'SVR': SVR(),
    'XGBR': XGBRegressor(),
    'KNNR': KNeighborsRegressor(),
    'MLPR': MLPRegressor()
}

params = {
    'RFM': {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]},
    'SVR': {'C': [1, 10], 'kernel': ['linear', 'rbf']},
    'XGBR': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]},
    'KNNR': {'n_neighbors': [5, 10], 'weights': ['uniform', 'distance']},
    'MLPR': {'hidden_layer_sizes': [(100,), (100, 50)], 'activation': ['relu', 'tanh'],'learning_rate_init': [0.001, 0.01],'max_iter': [500, 1000],'solver': ['adam', 'lbfgs']}
}

# Replace non-finite and NaN values with median
df_cleaned = replace_non_finite_with_median(df_replace_outliers)

# # Save original values for later
# df_cleaned['Original_PoliceStationCode'] = df_cleaned['PoliceStationCode']
# df_cleaned['Original_Quarter'] = df_cleaned['Quarter']
# df_cleaned['Original_ProvinceCode'] = df_cleaned['ProvinceCode']

# Prepare train/test data
X = df_cleaned[['2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']]
y = df_cleaned['2023']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Run scenarios
scenario_funcs = {
    "Scenario 1": scenario_1,
    "Scenario 2": scenario_2,
    "Scenario 3": scenario_3,
    "Scenario 4": scenario_4
}

for scenario_name, scenario_func in scenario_funcs.items():
    st.subheader(f"Results for {province_code_value} :  {scenario_name} ")
    
    for name in models.keys():
        scenario_predictions, scenario_metrics = train_and_predict(df_cleaned, {name: models[name]}, {name: params[name]}, scenario_func, scenario_name, df_cleaned)
        
        # Display predictions and metrics for each algorithm and scenario
        st.markdown(f"### Predictions for {name} under {scenario_name}")
        st.dataframe(scenario_predictions, height=300, use_container_width=True)
        
        st.markdown(f"### Metrics for {name} under {scenario_name}")
        st.dataframe(scenario_metrics.head(1), height=50, use_container_width=True)               

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

with RealTimePreditionsTab9:
    with st.expander(f'Trained scaled', expanded=False):
        st.header(f'Scaled :', divider='rainbow')
        st.dataframe(X_train_scaled, height=210, hide_index=True, use_container_width=True)

    # with st.expander(f'Predictions : With Encoding', expanded=False):
    #     st.header(f'Predictions and True values :', divider='rainbow')
    #     st.dataframe(df_wide, height=210, hide_index=True, use_container_width=True)

# with RealTimeMetricsTab10:
#     with st.expander(f'Mertics: Encoding', expanded=False):
#         st.header(f'Train :', divider='rainbow')
#         st.dataframe(final_predictions, height=210, hide_index=True, use_container_width=True)
#         st.header(f'Test :', divider='rainbow')
#         st.dataframe(final_metrics, height=210, hide_index=True, use_container_width=True)


