import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split , 
    GridSearchCV
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
    fetch_crime_data, 
    fetch_provinces_data, 
    fetch_policestation_data
    )
from modelResponse.modeltrainingResponse import (
    train_rfm_model, 
    train_ann_model,  
    train_knn_model, 
    train_svr_model, 
    train_xgb_model
    )
from modelResponse.outliersResponse import (
    identify_outliers, 
    replace_outliers
    )
from modelResponse.hyperparametersResponse import mse_gridSearchCV
from shapleyPostHocResponse.shapleyPostHocResopnse import display_shap_plots
from sklearn.metrics import ( 
    mean_absolute_error as meanae, 
    mean_squared_error as meanse,
    r2_score as r2score, 
    mean_absolute_percentage_error as meanape
)

from math import sqrt
import warnings
warnings.filterwarnings('ignore')

mae_train_values, mse_train_values, r2_train_values, mape_train_values = [], [], [], []
mae_test_values, mse_test_values, r2_test_values, mape_test_values = [], [], [], []

st.set_page_config(page_title='ML Model Building', page_icon='🤖', layout='wide')

st.title('Interpretable Regression ML Model Builder')

with st.expander('About this app'):
    st.markdown('**What can this app do?**')
    st.info('This app allows users to build a machine learning (ML) model in an end-to-end workflow. Particularly, this encompasses data upload, data pre-processing, ML model building and post-model analysis.')

    st.markdown('**How to use the app?**')
    st.warning('To engage with the app, go to the sidebar and 1. Select a data set and 2. Adjust the model parameters by adjusting the various slider widgets. This will initiate the ML model building process, display the model results, and allow users to download the generated models and accompanying data.')

    st.markdown('**Under the hood**')
    st.markdown('Data sets:')
    st.code('''- Drug solubility data set
    ''', language='markdown')
  
    st.markdown('Libraries used:')
    st.code('''- Pandas for data wrangling
               - Scikit-learn for building a machine learning model
               - Altair for chart creation
            - Streamlit for user interface
    ''', language='markdown')

with st.sidebar:
    st.header(f'1. Input data')

    df_crime_data_db = pd.DataFrame()
    df_identify_outliers = pd.DataFrame()
    df_replace_outliers = pd.DataFrame()
    df_provinces = pd.DataFrame()
    df_policestations = pd.DataFrame()

    inputdatatype = st.radio('Select input data type', options=['Use database data'], index=0)

    if inputdatatype == 'Use database data':
        st.markdown('**1.1. Use database data**')
        with st.expander('Select Input Parameters'):
            df_provinces = fetch_provinces_data()
            province_name = st.selectbox('Select Province', df_provinces['ProvinceName'], format_func=lambda x: x, index=8)
            province_code_value = df_provinces[df_provinces['ProvinceName'] == province_name]['ProvinceCode'].values[0]

            df_policestations = fetch_policestation_data(province_code_value)
            # Ensure the index is within the valid range
            valid_index = min(0, len(df_policestations) - 1)
            
            if province_code_value == 'ZA.WC':
                valid_index = 110

            police_station_name = st.selectbox('Select Police Station', df_policestations['StationName'], format_func=lambda x: x, index=valid_index)
            police_code_value = df_policestations[df_policestations['StationName'] == police_station_name]['StationCode'].values[0]

            year_mapping = st.slider('Select year range from 2016 - 2023', 2023, 2016)
            quarter = st.radio('Select quarter of year', options=[1, 2, 3, 4], index=0)

            df_crime_data_db = fetch_crime_data(province_code_value, police_code_value, year_mapping, quarter)

        st.markdown('**1.2. Identify outliers**')
        identify_outlier = st.toggle('Identify outliers')
        
        if identify_outlier:
            df_identify_outliers = identify_outliers(df_crime_data_db)

        st.markdown('**1.3. Replace outliers with median**')
        replace_outlier = st.toggle('Replace with Median')

        if replace_outlier:
            df_replace_outliers = replace_outliers(df_crime_data_db)

        st.markdown('**1.4. Set Test and Train Parameters**')
        parameter_split_size = st.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)


    st.subheader('2. Select Algorithm')
    with st.expander('Algorithms'):
        algorithm = st.radio('', options=['ANN (MLPRegressor)', 'KNN', 'RFM', 'SVR','XGBoost'], index=2)

    st.subheader('3. Learning Parameters')
    with st.expander('See parameters', expanded=False):
            parameter_n_estimators = st.slider('Number of estimators (n_estimators)', 0, 1000, 100, 100)
            parameter_max_features = st.select_slider('Max features (max_features)', options=['sqrt', 'log2', 'all']) #['all', 'sqrt', 'log2']
            parameter_min_samples_split = st.slider('Minimum number of samples required to split an internal node (min_samples_split)', 2, 10, 2, 1)
            parameter_min_samples_leaf = st.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

    st.subheader('4. General Parameters')
    with st.expander('See parameters', expanded=False):
        parameter_random_state = st.slider('Seed number (random_state)', 0, 1000, 42, 1)
        parameter_criterion = st.select_slider('Performance measure (criterion)', options=['squared_error', 'absolute_error', 'friedman_mse'])
        parameter_bootstrap = st.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
        parameter_oob_score = st.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])

    sleep_time = st.slider('Sleep time', 0, 3, 0)

if not df_crime_data_db.empty: 
    with st.status("Running ...", expanded=True) as status:
    
        st.write("Loading data ...")
        time.sleep(sleep_time)

        st.write("Preparing data ...")
        time.sleep(sleep_time)
       

       # Dictionary to store predicted values
        predictions_dict = {}
        crime_categories_list = []
        model_results_list = []

        # Initialize empty lists for metrics and crime categories
        crime_categories_list = df_crime_data_db['CrimeCategory'].tolist()

    for index, row in df_crime_data_db.iterrows():

        X = df_crime_data_db .drop(columns=['CrimeCategory'])  # Drop the CrimeCategory column
        crime_category = df_crime_data_db['CrimeCategory']   # Keep the CrimeCategory column separately

        #crime_categories_list.append(crime_category)
        #st.write(crime_category)

        # Define y as a numeric target variable, such as the mean across years for each category
        y = df_crime_data_db.iloc[:, 1:].mean(axis=1)  # Assuming y should be the mean across all years

        X_train, X_test, y_train, y_test, crime_category_train, crime_category_test = train_test_split(
        X, y, crime_category, test_size=(100-parameter_split_size)/100, random_state=parameter_random_state)

       # Reset indices to ensure proper alignment
        crime_category_train = crime_category_train.reset_index(drop=True)
        crime_category_test = crime_category_test.reset_index(drop=True)
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)

        # Insert CrimeCategory as the first column
        X_train_display = X_train.copy()
        X_train_display.insert(0, 'CrimeCategory', crime_category_train)  # Insert as the first column
        X_test_display = X_test.copy()
        X_test_display.insert(0, 'CrimeCategory', crime_category_test)  # Insert as the first column

        if algorithm == 'ANN (MLPRegressor)':
            model = MLPRegressor()
            model = train_ann_model(parameter_n_estimators, parameter_max_features, parameter_min_samples_split, parameter_min_samples_leaf, parameter_random_state, parameter_criterion, parameter_bootstrap, parameter_oob_score)
        
        elif algorithm == 'KNN':
            model = KNeighborsRegressor()
            model = train_knn_model(parameter_n_estimators, parameter_max_features, parameter_min_samples_split, parameter_min_samples_leaf, parameter_random_state, parameter_criterion, parameter_bootstrap, parameter_oob_score)

        elif algorithm == 'RFM':
            model = RandomForestRegressor()
            model = train_rfm_model(parameter_n_estimators, parameter_max_features, parameter_min_samples_split, parameter_min_samples_leaf, parameter_random_state, parameter_criterion, parameter_bootstrap, parameter_oob_score)
           
        elif algorithm == 'SVR':
            model = SVR()
            model = train_svr_model(parameter_n_estimators, parameter_max_features, parameter_min_samples_split, parameter_min_samples_leaf, parameter_random_state, parameter_criterion, parameter_bootstrap, parameter_oob_score)

        elif algorithm == 'XGBoost':
            model = XGBRegressor()
            model = train_xgb_model(parameter_n_estimators, parameter_max_features, parameter_min_samples_split, parameter_min_samples_leaf, parameter_random_state, parameter_criterion, parameter_bootstrap, parameter_oob_score)

        # Adjust the computation of max_features
        if parameter_max_features == 'all':
            parameter_max_features = None  # Use None for RandomForestRegressor to consider all features
            parameter_max_features_metric = X.shape[1]  # Number of features
        elif parameter_max_features == 'sqrt' or parameter_max_features == 'log2':
            parameter_max_features_metric = parameter_max_features  # Keep track of the metric used
        else:
            parameter_max_features_metric = int(parameter_max_features)  # Convert to integer if numeric

        model.fit(X_train, y_train)

        #st.write("Applying model to make predictions ...")
        #time.sleep(sleep_time)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
            
        #st.write("Calculating performance metrics ...")
        #time.sleep(sleep_time)

    def calculate_metrics(row):
        crime_category = row['CrimeCategory']
        X = row.drop('CrimeCategory')  # Drop the CrimeCategory column
        y_train = X.iloc[:-1].values   # Example: select the first 6 values as y_train
        y_test = X.iloc[:-1].values    # Example: select the first 6 values as y_test
    
        # Example predictions (replace with your actual model predictions)
        y_train_pred = y_train * 0.5
        y_test_pred = y_test * 0.7

        # Store predicted values in the dictionary
        predictions_dict[crime_category] = {'true_values': y_test, 'predicted_values': y_test_pred}
    
        # Calculate metrics
        mae_train = meanae(y_train, y_train_pred)
        mse_train = meanse(y_train, y_train_pred)
        r2_train = r2score(y_train, y_train_pred)
        mape_train = meanape(y_train, y_train_pred)
    
        mae_test = meanae(y_test, y_test_pred)
        mse_test = meanse(y_test, y_test_pred)
        r2_test = r2score(y_test, y_test_pred)
        mape_test = meanape(y_test, y_test_pred)
    
        # Return metrics as a Series including CrimeCategory
        return pd.Series({
            'CrimeCategory': crime_category,
            'Training MSE': mse_train,
            'Training R2': r2_train,
            'Training MAE': mae_train,
            'Training MAPE': mape_train,
            'Test MSE': mse_test,
            'Test R2': r2_test,
            'Test MAE': mae_test,
            'Test MAPE': mape_test
            })
            
    if parameter_criterion == 'squared_error':
            parameter_criterion_string = 'MSE'
    else:
        parameter_criterion_string = 'MAE'

        # Apply the function to each row of df_crime_data_db and create model_results DataFrame
    model_results = df_crime_data_db.apply(calculate_metrics, axis=1)

        # Set 'CrimeCategory' as the index
    model_results.set_index('CrimeCategory', inplace=True)
        
    status.update(label="Status", state="complete", expanded=False)

    st.header(f'Input data for {algorithm} algorithm', divider='rainbow')
    train_ratio = parameter_split_size
    test_ratio = 100 - parameter_split_size
    split_ration_value = f'{train_ratio} : {test_ratio}'
    split_ration = f'Split Ration % Train\:Test'
    col = st.columns(5)
    col[0].metric(label="No. of samples", value=X.shape[0], delta="")
    col[1].metric(label="No. of X variables", value=X.shape[1], delta="")
    col[2].metric(label="No. of Training samples", value=X_train.shape[0], delta="")
    col[3].metric(label="No. of Test samples", value=X_test.shape[0], delta="")
    col[4].metric(label= split_ration, value= split_ration_value, delta="")
        
    with st.expander('Initial dataset', expanded=True):
            st.dataframe(df_crime_data_db, height=210, use_container_width=True)

             # Prepare data for the graph
            df_outliers_melt = df_crime_data_db.melt(id_vars=['CrimeCategory'], var_name='Year', value_name='Percentage')
            df_outliers_melt = df_outliers_melt.sort_values(by=['CrimeCategory', 'Year'])

            # Create the graph using seaborn
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=df_outliers_melt, x='Year', y='Percentage', hue='CrimeCategory', marker='o')
            plt.title('Crime Trends Over the Years')
            plt.xlabel('Year')
            plt.ylabel('Percentage')
            plt.legend(title='Crime Category', bbox_to_anchor=(1.05, 1), loc='upper left')

            # Annotate each data point with its value
            for i in range(df_outliers_melt.shape[0]):
                plt.text(df_outliers_melt['Year'].iloc[i], df_outliers_melt['Percentage'].iloc[i], 
                    f"{df_outliers_melt['Percentage'].iloc[i]:.2f}", color='black', ha='right', va='bottom')

            plt.xticks(rotation=45)

            # Display the graph in Streamlit
            st.pyplot(plt)

    if not df_identify_outliers.empty:
        with st.expander('Identify outliers', expanded=True):
            performance_col = st.columns((2, 0.2, 3))

            with performance_col[0]:
                st.header('Outliers', divider='rainbow')
                st.dataframe(df_identify_outliers)

            # Plot box plot of the data with outliers replaced   
            with performance_col[2]:
                st.header('Outliers percentage plot', divider='rainbow')
                plt.figure(figsize=(12, 8))
                sns.boxplot(x="Year", y="Percentage", data=df_identify_outliers)
                plt.title("Box plot identifying the outliers")
                plt.xticks(rotation=45)

                # Add annotations
                for i, row in df_identify_outliers.iterrows():  # range(len(df_identify_outliers)):
                    plt.text(
                        x=i % len(df_identify_outliers['Year'].unique()), 
                        y=row['Percentage'] + 2, # df_identify_outliers['Percentage'][i] + 2,  # Adjust the position to avoid overlap with the box plot
                        s=f"{df_identify_outliers['Percentage'][i]:.1f}", 
                        fontsize=9,
                        color='black',
                        ha='center'
                    )

                st.pyplot(plt)

    
    #df_replace_outliers
    if not (df_replace_outliers.empty and df_identify_outliers.empty): 
        with st.expander('Replaced outliers by the median', expanded=True):
            st.dataframe(df_replace_outliers, height=210, use_container_width=True)
            
            if not (df_replace_outliers.empty):
                df_outliers_replaced_melt = df_replace_outliers.melt(id_vars=['CrimeCategory'], var_name='Year', value_name='Percentage')
                

                # Create the graph using seaborn
                plt.figure(figsize=(10, 6))
                sns.lineplot(data=df_outliers_replaced_melt, x='Year', y='Percentage', hue='CrimeCategory', marker='o')
                plt.title('Crime Trends Over the Years with Outliers Replaced ')
                plt.xlabel('Year')
                plt.ylabel('Percentage')
                plt.legend(title='Crime Category', bbox_to_anchor=(1.05, 1), loc='upper left')


            # Annotate each data point with its value
                for i in range(df_outliers_replaced_melt.shape[0]):
                    plt.text(df_outliers_replaced_melt['Year'].iloc[i], df_outliers_replaced_melt['Percentage'].iloc[i], 
                        f"{df_outliers_replaced_melt['Percentage'].iloc[i]:.2f}", color='black', ha='right', va='bottom')

                plt.xticks(rotation=45)

                # Display the graph in Streamlit
                st.pyplot(plt)

   # Display the updated train and test splits
    with st.expander('Train split', expanded=False):
        train_col = st.columns((3, 1))
        with train_col[0]:
            st.markdown('**X**')
            st.dataframe(X_train_display, height=210, hide_index=True, use_container_width=True)
        with train_col[1]:
            st.markdown('**y**')
            st.dataframe(y_train, height=210, hide_index=True, use_container_width=True)

    with st.expander('Test split', expanded=False):
        test_col = st.columns((3, 1))
        with test_col[0]:
            st.markdown('**X**')
            st.dataframe(X_test_display, height=210, hide_index=True, use_container_width=True)
        with test_col[1]:
            st.markdown('**y**')
            st.dataframe(y_test, height=210, hide_index=True, use_container_width=True)

    df_crime_data_db.to_csv('dataset.csv', index=False)
    X_train.to_csv('X_train.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)
        
    list_files = ['dataset.csv', 'X_train.csv', 'y_train.csv', 'X_test.csv', 'y_test.csv']
    with zipfile.ZipFile('dataset.zip', 'w') as zipF:
        for file in list_files:
            zipF.write(file, compress_type=zipfile.ZIP_DEFLATED)

    with open('dataset.zip', 'rb') as datazip:
        btn = st.download_button(
                label='Download ZIP',
                data=datazip,
                file_name="dataset.zip",
                mime="application/octet-stream"
                )
        
    st.header(f'{algorithm} model parameters', divider='rainbow')
    parameters_col = st.columns(3)
    parameters_col[0].metric(label="Data split ratio (% for Training Set)", value=parameter_split_size, delta="")
    parameters_col[1].metric(label="Number of estimators (n_estimators)", value=parameter_n_estimators, delta="")
    parameters_col[2].metric(label="Max features (max_features)", value=parameter_max_features_metric, delta="")
    #parameters_col[4].metric(label="Min features (min_leaf)", value=parameter_min_samples_leaf, delta="")
        
    importances = model.feature_importances_
    feature_names = list(X.columns)
    forest_importances = pd.Series(importances, index=feature_names)
    df_importance = forest_importances.reset_index().rename(columns={'index': 'feature', 0: 'value'})
        
    bars = alt.Chart(df_importance).mark_bar(size=40).encode(
            x='value:Q',
            y=alt.Y('feature:N', sort='-x')
        ).properties(height=250)

    performance_col = st.columns((2, 0.2, 3))
    with performance_col[0]:
        st.header('Model performance', divider='rainbow')
        #st.write('Model performance to be edited for now it is hidden')
        st.dataframe(model_results.T.reset_index().rename(columns={'index': 'Parameter', 0: 'Value'}))
    with performance_col[2]:
        st.header('Feature importance', divider='rainbow')
        st.altair_chart(bars, theme='streamlit', use_container_width=True)

    st.header(f'{algorithm} prediction results', divider='rainbow')
    s_y_train = pd.Series(y_train, name='True Trained').reset_index(drop=True)
    s_y_train_pred = pd.Series(y_train_pred, name='Predicted Trained').reset_index(drop=True)
    #df_train = pd.DataFrame(data=[s_y_train, s_y_train_pred], index=None).T
    df_train = pd.DataFrame({'CrimeCategory': crime_category_train, 'True Trained': s_y_train, 'Predicted Trained': s_y_train_pred})
    df_train['class'] = 'train'
    s_y_test = pd.Series(y_test, name='actual').reset_index(drop=True)
    s_y_test_pred = pd.Series(y_test_pred, name='predicted').reset_index(drop=True)
    #df_test = pd.DataFrame(data=[s_y_test, s_y_test_pred], index=None).T
    df_test = pd.DataFrame({'CrimeCategory': crime_category_test, 'actual': s_y_test, 'predicted': s_y_test_pred})
    df_test['class'] = 'test'
        
    df_prediction = pd.concat([df_train, df_test], axis=0)
        
    prediction_col = st.columns((2, 0.2, 3))
        
    with prediction_col[0]:
        st.dataframe(df_prediction, height=320, use_container_width=True)

    with prediction_col[2]:
        scatter = alt.Chart(df_prediction).mark_circle(size=60).encode(
                        x='actual',
                        y='predicted',
                           color='class'
                )
        st.altair_chart(scatter, theme='streamlit', use_container_width=True)

    st.header(f'{algorithm} Shapley values', divider='rainbow')
    with st.expander('Shapley values'):
        print('empty space')
    #     # Function to preprocess the data and get the XGBoost model prediction
    # # def model_predict(X):
    # #     return  best_mlp_model.predict(X) 
 
       
    # Predict
    model_predict = model.predict(X_test)

    # Explainer using Kernel SHAP and the background dataset
    explainer = shap.KernelExplainer(model.predict, X_train)

    # # Run initjs()
    # shap.initjs()

    # # Summary plot for each feature's impact on the output
    # shap_values = explainer.shap_values(X_test)
    # shap.summary_plot(shap_values, X_test, feature_names=X_test.columns)

    # # Individual SHAP value plot for a specific instance
    # instance_index = 0  # You can choose any instance index
    # shap.force_plot(explainer.expected_value, shap_values[instance_index, :], X_test.iloc[instance_index, :])

    # display_shap_plots(rf, X_train, X)

else:
    st.warning('👈 Upload a CSV file or click *"Load example data"* to get started!')
