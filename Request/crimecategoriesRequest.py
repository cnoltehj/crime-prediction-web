import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import altair as alt
import time

# Function to fetch data (simulated for demonstration)
def fetch_data(provincecode, policestationcode, year, quarter):
    # Simulated data fetching process
    crime_categories = [
        'Contact crime (Crimes against the person)',
        'TRIO Crime',
        'Contact-related crime',
        'Property-related crime',
        'Other serious crime',
        '17 Community reported serious crime',
        'Crime detected as a result of police action'
    ]
    
    # Simulated data for each crime category over the years
    data = {
        'CrimeCategory': crime_categories,
        2017: np.random.uniform(low=-50, high=50, size=len(crime_categories)),
        2018: np.random.uniform(low=-50, high=50, size=len(crime_categories)),
        2019: np.random.uniform(low=-50, high=50, size=len(crime_categories)),
        2020: np.random.uniform(low=-50, high=50, size=len(crime_categories)),
        2021: np.random.uniform(low=-50, high=50, size=len(crime_categories)),
        2022: np.random.uniform(low=-50, high=50, size=len(crime_categories)),
        2023: np.random.uniform(low=-50, high=50, size=len(crime_categories)),
    }
    
    # Convert data dictionary to a DataFrame
    df = pd.DataFrame(data)
    
    # Set CrimeCategory as the index
    df.set_index('CrimeCategory', inplace=True)
    
    return df

# Page title and configuration
st.set_page_config(page_title='Crime Hotspot Interpretable ML Model Building', page_icon='🤖', layout='wide')
st.title('🤖 Crime Hotspot Interpretable ML Model Building')

# Sidebar for accepting input parameters
st.sidebar.header('Input Parameters')

# Example inputs (replace with actual input widgets)
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
provincecode = st.sidebar.selectbox('Select Province', list(province_mapping.keys()))

policestation_mapping = {
    'Parow': 'PRW104WC',
    'Athlone': 'AT03WC',
    'Bellville': 'BV08WC',
    'Bishop Lavis': 'BL010WC',
    'Gugulethu': 'GUG49WC',
    'Khayelitsha': 'KHL57WC',
    'Manenberg': 'MBG83WC',
}
policestationcode = st.sidebar.selectbox('Select Police Station', list(policestation_mapping.keys()))

year = st.sidebar.slider('Year', 2016, 2023, 2020)
quarter = st.sidebar.radio('Quarter', [1, 2, 3, 4], index=0)

if st.sidebar.button('Fetch Data'):
    st.sidebar.markdown('Fetching data...')
    data = fetch_data(provincecode, policestation_mapping[policestationcode], year, quarter)
    st.session_state['data'] = data

# Main content panel
st.header('Input Data Overview')

if 'data' in st.session_state:
    data = st.session_state['data']
    st.subheader('Fetched Data:')
    st.dataframe(data)

    st.header('3. Set Parameters')
    parameter_split_size = st.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

    st.subheader('3.1. Learning Parameters')
    with st.expander('See parameters'):
        parameter_n_estimators = st.selectbox('Number of estimators (n_estimators)', range(0, 1001, 100), index=10)
        parameter_max_features = st.selectbox('Max features (max_features)', ['auto', 'sqrt', 'log2'])
        parameter_min_samples_split = st.selectbox('Minimum number of samples required to split an internal node (min_samples_split)', range(2, 11), index=1)
        parameter_min_samples_leaf = st.selectbox('Minimum number of samples required to be at a leaf node (min_samples_leaf)', range(1, 11), index=1)

    st.subheader('3.2. General Parameters')
    with st.expander('See parameters', expanded=False):
        parameter_random_state = st.slider('Seed number (random_state)', 0, 1000, 42, 1)
        parameter_criterion = st.select_slider('Performance measure (criterion)', options=['mse', 'mae'])
        parameter_bootstrap = st.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
        parameter_oob_score = st.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])

    sleep_time = st.slider('Sleep time', 0, 3, 0)

    # Prepare data for modeling
    df = data.reset_index()
    X = df.drop(columns=['CrimeCategory'])
    y = X.pop(2023)  # Assuming the target variable is the year 2023

    st.write("Splitting data ...")
    time.sleep(sleep_time)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100-parameter_split_size)/100, random_state=parameter_random_state)

    st.write("Model training ...")
    time.sleep(sleep_time)

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

    st.write("Evaluating performance metrics ...")
    time.sleep(sleep_time)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    st.subheader('Model Performance')
    st.write(f"Train MSE: {train_mse:.3f}")
    st.write(f"Train R^2: {train_r2:.3f}")
    st.write(f"Test MSE: {test_mse:.3f}")
    st.write(f"Test R^2: {test_r2:.3f}")

    # Altair chart for comparing predicted vs actual
    st.subheader('Prediction vs Actual')
    chart_data = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred})
    chart = alt.Chart(chart_data).mark_circle(size=60).encode(
        x='Actual',
        y='Predicted',
        tooltip=['Actual', 'Predicted']
    ).interactive()

    st.altair_chart(chart, use_container_width=True)
else:
    st.warning('👈 Click *"Fetch Data"* in the sidebar to get started!')
