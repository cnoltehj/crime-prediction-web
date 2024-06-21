import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import altair as alt
import time
import zipfile
from Request.crimecategoriesRequest import fetch_data

st.set_page_config(page_title='ML Model Building', page_icon='🤖', layout='wide')

st.title('🤖 ML Model Building')

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
    st.header('1.1. Input data')

    df_db = pd.DataFrame()

    inputdatatype = st.radio('Select input data type', options=['Use database data'], index=0)

    if inputdatatype == 'Use database data':
        st.markdown('**1. Use database data**')
        with st.expander('Select Input Parameters'):
            province_mapping = {
                'Western Cape': 'ZA.WC',
                'Eastern Cape': 'ZA.EC',
                'Free State': 'ZA.FS',
                'Gauteng': 'ZA.GP',
                'Mpumalanga': 'ZA.MP',
                'Northern Cape': 'ZA.NC',
                'KwaZulu-Natal': 'ZA.NL',
                'Limpopo': 'ZA.NP',
                'North-West': 'ZA.NW',
            }
            provincecode = st.selectbox('Select Province', options=list(province_mapping.keys()), format_func=lambda x: x, index=0)
            provincecode_value = province_mapping[provincecode]

            policestation_mapping = {
                'Parow': 'PRW104WC',
                'Athlone': 'AT03WC',
                'Bellville': 'BV08WC ',
                'Bishop Lavis': 'BL010WC',
                'Gugulethu': 'GUG49WC',
                'Khayelitsha': 'KHL57WC',
                'Manenberg': 'MBG83WC'
            }
            policestationcode = st.selectbox('Select Police Station', options=list(policestation_mapping.keys()), format_func=lambda x: x, index=0)
            policestationcode_value = policestation_mapping[policestationcode]

            year_mapping = st.slider('Select year range from 2016 - 2023', 2023, 2016)
            quarter = st.radio('Select quarter of year', options=[1, 2, 3, 4], index=0)

            df_db = fetch_data(provincecode_value, policestationcode_value, year_mapping, quarter)

    st.header('2. Set Parameters')
    parameter_split_size = st.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

    st.subheader('2.1. Learning Parameters')
    with st.expander('See parameters'):
        parameter_n_estimators = st.slider('Number of estimators (n_estimators)', 0, 1000, 100, 100)
        parameter_max_features = st.select_slider('Max features (max_features)', options=['all', 'sqrt', 'log2'])
        parameter_min_samples_split = st.slider('Minimum number of samples required to split an internal node (min_samples_split)', 2, 10, 2, 1)
        parameter_min_samples_leaf = st.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

    st.subheader('2.2. General Parameters')
    with st.expander('See parameters', expanded=False):
        parameter_random_state = st.slider('Seed number (random_state)', 0, 1000, 42, 1)
        parameter_criterion = st.select_slider('Performance measure (criterion)', options=['squared_error', 'absolute_error', 'friedman_mse'])
        parameter_bootstrap = st.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
        parameter_oob_score = st.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])

    sleep_time = st.slider('Sleep time', 0, 3, 0)

if not df_db.empty: 
    with st.status("Running ...", expanded=True) as status:
    
        st.write("Loading data ...")
        time.sleep(sleep_time)

        st.write("Preparing data ...")
        time.sleep(sleep_time)
        
      # Separate features and target
        X = df_db.drop(columns=['CrimeCategory'])  # Drop the CrimeCategory column
        # Define y as a numeric target variable, such as the mean across years for each category
        y = df_db.iloc[:, 1:].mean(axis=1)  # Assuming y should be the mean across all years

        # Now, y will contain numeric values like [-3.128, -8.8, -0.985, -32.121, 23.413, -5.915]

        # Proceed with splitting and model training
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100-parameter_split_size)/100, random_state=parameter_random_state)

        # Adjust the computation of max_features
        if parameter_max_features == 'all':
            parameter_max_features = None  # Use None for RandomForestRegressor to consider all features
            parameter_max_features_metric = X.shape[1]  # Number of features
        elif parameter_max_features == 'sqrt' or parameter_max_features == 'log2':
            parameter_max_features_metric = parameter_max_features  # Keep track of the metric used
        else:
            parameter_max_features_metric = int(parameter_max_features)  # Convert to integer if numeric

        # Model initialization with RandomForestRegressor
        rf = RandomForestRegressor(
            n_estimators=parameter_n_estimators,
            max_features=parameter_max_features,
            min_samples_split=parameter_min_samples_split,
            min_samples_leaf=parameter_min_samples_leaf,
            random_state=parameter_random_state,
            criterion=parameter_criterion,
            bootstrap=parameter_bootstrap,
            oob_score=parameter_oob_score)
        rf.fit(X_train, y_train)

        st.write("Applying model to make predictions ...")
        time.sleep(sleep_time)
        y_train_pred = rf.predict(X_train)
        y_test_pred = rf.predict(X_test)
            
        st.write("Calculating performance metrics ...")
        time.sleep(sleep_time)
        mse_train = mean_squared_error(y_train, y_train_pred)
        r2_train = r2_score(y_train, y_train_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)
        r2_test = r2_score(y_test, y_test_pred)
            
        if parameter_criterion == 'squared_error':
            parameter_criterion_string = 'MSE'
        else:
            parameter_criterion_string = 'MAE'
                
        rf_results = pd.DataFrame([mse_train, r2_train, mse_test, r2_test], index=[f'Training {parameter_criterion_string}', 'Training R2', f'Test {parameter_criterion_string}', 'Test R2'])
            
        for col in rf_results.columns:
            rf_results[col] = pd.to_numeric(rf_results[col], errors='ignore')
        rf_results = rf_results.round(3)
        
    status.update(label="Status", state="complete", expanded=False)

    st.header('Input data', divider='rainbow')
    col = st.columns(4)
    col[0].metric(label="No. of samples", value=X.shape[0], delta="")
    col[1].metric(label="No. of X variables", value=X.shape[1], delta="")
    col[2].metric(label="No. of Training samples", value=X_train.shape[0], delta="")
    col[3].metric(label="No. of Test samples", value=X_test.shape[0], delta="")
        
    with st.expander('Initial dataset', expanded=True):
            st.dataframe(df_db, height=210, use_container_width=True)
    with st.expander('Train split', expanded=False):
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

    df_db.to_csv('dataset.csv', index=False)
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
        
    st.header('Model parameters', divider='rainbow')
    parameters_col = st.columns(3)
    parameters_col[0].metric(label="Data split ratio (% for Training Set)", value=parameter_split_size, delta="")
    parameters_col[1].metric(label="Number of estimators (n_estimators)", value=parameter_n_estimators, delta="")
    parameters_col[2].metric(label="Max features (max_features)", value=parameter_max_features_metric, delta="")
        
    importances = rf.feature_importances_
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
        st.dataframe(rf_results.T.reset_index().rename(columns={'index': 'Parameter', 0: 'Value'}))
    with performance_col[2]:
        st.header('Feature importance', divider='rainbow')
        st.altair_chart(bars, theme='streamlit', use_container_width=True)

    st.header('Prediction results', divider='rainbow')
    s_y_train = pd.Series(y_train, name='actual').reset_index(drop=True)
    s_y_train_pred = pd.Series(y_train_pred, name='predicted').reset_index(drop=True)
    df_train = pd.DataFrame(data=[s_y_train, s_y_train_pred], index=None).T
    df_train['class'] = 'train'
            
    s_y_test = pd.Series(y_test, name='actual').reset_index(drop=True)
    s_y_test_pred = pd.Series(y_test_pred, name='predicted').reset_index(drop=True)
    df_test = pd.DataFrame(data=[s_y_test, s_y_test_pred], index=None).T
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

else:
    st.warning('👈 Upload a CSV file or click *"Load example data"* to get started!')
