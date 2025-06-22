import streamlit as st
import pandas as pd
import numpy as np
import requests
import altair as alt
from sklearn.model_selection import (
    train_test_split
    , KFold
    , cross_val_score
    , cross_validate
    )
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
import altair as alt
from sklearn.multioutput import MultiOutputRegressor
import time
import zipfile
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import shap
from dataRequest.crimedbRequest import (
    fetch_all_provinces,
    fetch_policestation_per_provinces,
    fetch_all_stats_province_quarterly
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


with st.sidebar:
    st.header(f'1. Input data')
    # with st.expander('Select Input Parameters'):
    df_provinces = fetch_all_provinces()
    province_name = st.selectbox('Select Province', df_provinces['ProvinceName'], format_func=lambda x: x, index=8)
    province_code_value = df_provinces[df_provinces['ProvinceName'] == province_name]['ProvinceCode'].values[0]

    #                 # TODO add for all other provinces

    #         if province_code_value == 'ZA.WC': #
    #             valid_index = 110

    df_policestations = fetch_policestation_per_provinces(province_code_value)
    #                 # Ensure the index is within the valid range
    #             valid_index = min(0, len(df_policestations) - 1)
    police_station_name = st.selectbox('Select Police Station', df_policestations['StationName'], format_func=lambda x: x, index=2)
    police_code_value = df_policestations[df_policestations['StationName'] == police_station_name]['StationCode'].values[0]
    year_mapping = st.slider('Select year range from 2017 - 2024', 2024, 2017)
    
    quarter_value = st.selectbox('Select Quarter', ['Q1', 'Q2', 'Q3', 'Q4'], index=0)
                
    st.markdown('**1.1. Split Dataset**')
    train_split = st.slider('Data split ratio (% for Training Set)', 60, 80, 60)
    validation_split = st.slider('Data split ratio (% for Validation Set)', 10, 20, 20)
    test_split = st.slider('Data split ratio (% for Test Set)', 10, 20, 20)

    # Check if the total equals 100
    total_split = train_split + validation_split + test_split
    if total_split != 100:
        st.error(f"The sum of Train ({train_split}%) + Validation ({validation_split}%) + Test ({test_split}%) must equal 100%. Current total: {total_split}%")
    else:
        st.success(f"Data split is valid: Train={train_split}%, Validation={validation_split}%, Test={test_split}%")

    #     st.subheader('3. Learning Parameters')
    #     with st.expander('See parameters', expanded=False):
    #             sleep_time = st.slider('Sleep time', 0, 3, 0)

    # if not df_crime_data_db.empty:
    #     with st.status("Running ...", expanded=True) as status:

    #             # st.write("Loading data ...")
    #             # time.sleep(sleep_time)

    #             # st.write("Preparing data ...")
    #             # time.sleep(sleep_time)

    #             # Initialize empty lists for metrics and crime categories
    #             crime_categories_list = df_crime_data_db['CrimeCategory'].tolist()

st.title('Expirement Interpretable Crime Hotspot Prediction')

AboutTab1,DataExtractionViewTab2,Transformationtab3,ModelTrainingPredictionTab4, ModelTrainingValidationTab5,ModelTrainingMerticsTab6,PostHocAnalysisTab7 = st.tabs(['About','ETL-Extra Data','Transform-Transformation','Model-Train-Prediction', 'Model-Train_Validation','Model-TrainMetrics','PostHocAnalysis'])

with AboutTab1:
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



        
with DataExtractionViewTab2:
        
        st.header(f'ETL - Extract Transform Load', divider='rainbow')

        # Fetch the prediction data
        print(f'Fetching data for {province_name} Province')
        print(fetch_all_stats_province_quarterly(province_code_value,quarter_value))

        df_fetch_all_stats_province_quarterly = pd.DataFrame(fetch_all_stats_province_quarterly(province_code_value,quarter_value))

        st.dataframe(df_fetch_all_stats_province_quarterly) 

        # st.header(f'{province_name}: Predictions vs True_Value', divider='rainbow')
        # df_prediction_ui = pd.DataFrame(def_fetch_stats_province_policestation_quarterly_algorithm_db)

        # performance_col = st.columns((10, 0.8, 0.8))

        # with performance_col[0]:       
        #     # Bar plot for Prediction vs True_Value
        #     bar_width = 0.35
        #     index = range(len(df_prediction_ui))

        #             # Create the bar plot
        #     fig, ax = plt.subplots(figsize=(5, 2)) # Adjust width=8 and height=4 as needed
        #     bar1 = ax.bar(index, df_prediction_ui['Prediction'], bar_width, label='Prediction', color='b')
        #     bar2 = ax.bar([i + bar_width for i in index], df_prediction_ui['True_Value'], bar_width, label='True Value', color='r')

        #     ax.set_xlabel('CrimeCategory')
        #     ax.set_ylabel('Counts')
        #     ax.set_title(f'{province_code_value}: {police_station_name} : Prediction vs True Value')
        #     ax.set_xticks([i + bar_width / 2 for i in index])
        #     ax.set_xticklabels(df_prediction_ui['CrimeCategory'], rotation=45, ha='right',fontsize = 5)
        #     ax.legend(fontsize = 5)

        #     # Add the values on top of the bars
        #     for i, v in enumerate(df_prediction_ui['Prediction']):
        #         ax.text(i, v + 5, f'{v:.0f} ', ha='center', va='bottom', fontsize=4)  # Add value for Prediction

        #     for i, v in enumerate(df_prediction_ui['True_Value']):
        #         ax.text(i + bar_width, v + 5, f': {v:.0f}', ha='center', va='bottom', fontsize=4)  # Add value for True_Value


        #             # Display the bar plot in Streamlit
        #     st.pyplot(fig)

        #         #st.header('Predictions plot', divider='rainbow')
        #         # Header for the first section
        # st.header(f'{province_code_value}: {police_station_name} Police Station : Predictions', divider='rainbow')
        # st.dataframe(def_fetch_stats_province_policestation_quarterly_algorithm_db) #.sort_values(by='CrimeCategory'))

        # st.header(f'{province_name}: {police_station_name} Police station Initial dataset', divider='rainbow')
        # if not def_fetch_stats_province_policestation_quarterly_algorithm_db.empty:
        #     st.dataframe(def_fetch_stats_province_policestation_quarterly_algorithm_db)  # Show the dataframe
        # else:
        #     st.write("No data available or API returned an empty result.")

        # st.header(f'{province_name}: All initial dataset used for training models', divider='rainbow')
        # st.dataframe(df_suggeted_province_quarterly_data_db) #.sort_values(by='PoliceStationCode'))

        # # # Display the heatmap for numerical features in the dataframe
        # # st.header(f'{province_name}: Heatmap of correlation between numerical features')

        # # # Create a correlation matrix
        # # correlation_matrix = df_suggeted_province_quarterly_data_db.corr()

        # # # Plot the heatmap
        # # fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the size of the heatmap as needed
        # # sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax, linewidths=0.5)

        # # # Display the heatmap in Streamlit
        # # st.pyplot(fig)
    

with Transformationtab3:
#         st.header('Identify outliers', divider='rainbow')
#         performance_col = st.columns((2, 0.2, 3))
        st.header(f'Replace None and treat as Nan', divider='rainbow')
        # Convert None to np.nan
        df_fetch_all_stats_province_quarterly.replace({None: np.nan}, inplace=True)

        # Then fill NaN values using forward fill and backfill
        df_fetch_all_stats_province_quarterly.fillna(method='ffill', inplace=True)
        df_fetch_all_stats_province_quarterly.fillna(method='bfill', inplace=True)

        # Display the DataFrame after replacing None and treating as NaN
        st.dataframe(df_fetch_all_stats_province_quarterly)

        year_cols = [str(y) for y in range(2016, 2024)]

        # Ensure DataFrame has those columns
        year_cols = [c for c in year_cols if c in df_fetch_all_stats_province_quarterly.columns]

        # Now df will have:
        # ['CrimeCategory', 'ProvinceCode', 'PoliceStationCode', 'QuarterofYear'] + year_cols

        performance_col = st.columns(2)

        with performance_col[0]:
            st.header('Outliers', divider='rainbow')
            df_out = identify_outliers_data(df_fetch_all_stats_province_quarterly)
            st.dataframe(df_out.sort_values(by='PoliceStationCode'))

        with performance_col[1]:
            st.header('Outliers Percentage Plot', divider='rainbow')

            df_out = identify_outliers_data(df_fetch_all_stats_province_quarterly)

            # Melt using only the explicit year columns
            df_melted = df_out.melt(
                id_vars=['CrimeCategory', 'ProvinceCode', 'PoliceStationCode', 'Quarter', 'Outliers'],
                value_vars=year_cols,
                var_name='Year',
                value_name='Percentage'
            )

            def parse_outliers(o):
                if isinstance(o, str):
                    return [float(x) for x in o.split(',') if x.strip()]
                if isinstance(o, (int, float)):
                    return [float(o)]
                return []

            df_melted['Outliers'] = df_melted['Outliers'].apply(parse_outliers)
            df_exploded = df_melted.explode('Outliers')

            fig, ax = plt.subplots(figsize=(12, 8))
            sns.boxplot(x='Year', y='Percentage', data=df_melted, ax=ax)
            ax.set_title("Box Plot Identifying the Outliers")
            ax.tick_params(axis='x', rotation=45)

            y_min, y_max = ax.get_ylim()
            # Annotate outliers
            for idx, row in df_exploded.iterrows():
                val = row['Outliers']
                if not pd.isna(val):
                    y_text = min(val + 2, y_max - 5)
                    ax.text(
                        x=year_cols.index(row['Year']),  # use index of the year
                        y=y_text,
                        s=f"{val:.1f}",
                        fontsize=9,
                        color='black',
                        ha='center'
                    )

            st.pyplot(fig)

        st.header('Replaced outliers with the median value', divider='rainbow')
        df_replaced = replace_outliers_data(df_fetch_all_stats_province_quarterly)
        st.dataframe(df_replaced.sort_values(by='PoliceStationCode'), use_container_width=True)

        years = list(range(2016, 2023))  # 2016 through 2023

        for year in years:
            # 1. Build pivot for the given year
            df_pivot = df_replaced.pivot_table(
                index='CrimeCategory',
                columns='PoliceStationCode',
                values=str(year),
                aggfunc='sum'
            )

            # 2. Create a clustered heatmap (clustermap)
            #    Note: sns.clustermap returns a ClusterGrid object
            cg = sns.clustermap(
                df_pivot,
                method='ward',
                metric='euclidean',
                cmap='coolwarm',
                figsize=(10, 5),
                standard_scale=0  # scale each row
            )

            # 3. Add dynamic title
            cg.fig.suptitle(f"Clustered Heatmap of Crime by Station ({year})", y=1.02)

            # 4. Render in Streamlit
            st.pyplot(cg.fig)


          
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

with ModelTrainingPredictionTab4:
        st.header(f'Model Training Predictions', divider='rainbow')

        # 1. Define models & grids
        models = {
            'RFM': RandomForestRegressor(),
            'SVR': SVR(),
            'XGBR': XGBRegressor(),
            'KNNR': KNeighborsRegressor(),
            'MLPR': MLPRegressor(max_iter=1000)
        }
        params = {
            'RFM':   {'n_estimators': [100,200], 'max_depth':[None,10,20]},
            'SVR':   {'C':[1,10],'kernel':['linear','rbf']},
            'XGBR':  {'n_estimators':[100,200],'learning_rate':[0.01,0.1],'max_depth':[3,5]},
            'KNNR':  {'n_neighbors':[5,10],'weights':['uniform','distance']},
            'MLPR':  {'hidden_layer_sizes':[(100,),(100,50)],'activation':['relu','tanh'],
                    'learning_rate_init':[0.001,0.01],'solver':['adam','lbfgs']}
        }

        
        def scenario_1(df):
            le_psc = LabelEncoder()
            le_qtr = LabelEncoder()
            df['PoliceStationCode'] = le_psc.fit_transform(df['PoliceStationCode'])
            df['Quarter'] = le_qtr.fit_transform(df['Quarter'])
            return df, le_psc, le_qtr

        def scenario_2(df):
            onehot_encoder = OneHotEncoder(sparse_output=False)
            encoded_features = onehot_encoder.fit_transform(df[['PoliceStationCode', 'Quarter']])
            encoded_df = pd.DataFrame(encoded_features, columns=onehot_encoder.get_feature_names_out(['PoliceStationCode', 'Quarter']))
            df = pd.concat([df, encoded_df], axis=1).drop(['PoliceStationCode', 'Quarter'], axis=1)
            return df, None, None

        def scenario_3(df):
            le_psc = LabelEncoder()
            df['PoliceStationCode'] = le_psc.fit_transform(df['PoliceStationCode'])
            onehot_encoder = OneHotEncoder(sparse_output=False)
            encoded_features = onehot_encoder.fit_transform(df[['Quarter']])
            encoded_df = pd.DataFrame(encoded_features, columns=onehot_encoder.get_feature_names_out(['Quarter']))
            df = pd.concat([df, encoded_df], axis=1).drop(['Quarter'], axis=1)
            return df, le_psc, None

        def scenario_4(df):
            le_qtr = LabelEncoder()
            df['Quarter'] = le_qtr.fit_transform(df['Quarter'])
            onehot_encoder = OneHotEncoder(sparse_output=False)
            encoded_features = onehot_encoder.fit_transform(df[['PoliceStationCode']])
            encoded_df = pd.DataFrame(encoded_features, columns=onehot_encoder.get_feature_names_out(['PoliceStationCode']))
            df = pd.concat([df, encoded_df], axis=1).drop(['PoliceStationCode'], axis=1)
            return df, None, le_qtr

        def scenario_5(df):
            le_psc = LabelEncoder()
            le_qtr = LabelEncoder()
            df['PoliceStationCode'] = le_psc.fit_transform(df['PoliceStationCode'])
            df['Quarter'] = le_qtr.fit_transform(df['Quarter'])
            return df, le_psc, le_qtr


        def compute_metrics(y_true, y_pred):
            return {
                'MAE': mean_absolute_error(y_true, y_pred),
                'MSE': mean_squared_error(y_true, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
                'R2': r2_score(y_true, y_pred),
                'MAPE': mean_absolute_percentage_error(y_true, y_pred)
            }

        def run_recursive_forecast(df, feature_cols, forecast_years, scenario_func):
            df = df.copy()
            df['Quarter_Original'] = df['Quarter']  # Preserve original for display
            df_encoded, le_psc, le_qtr = scenario_func(df.copy()) if 'scenario' in scenario_func.__name__ else (df.copy(), None, None)

            results = {}

            # GLOBAL split
            X_full = df_encoded[feature_cols].values.astype(float)
            y_full = df_encoded[feature_cols[-1]].values.astype(float)

            X_temp, X_test, y_temp, y_test = train_test_split(X_full, y_full, test_size=0.20, random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.20

            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)

            for model_name, model in models.items():
                try:
                    # GridSearchCV + KFold
                    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
                    gs = GridSearchCV(model, params[model_name], cv=kfold, scoring='neg_mean_squared_error', n_jobs=-1)
                    gs.fit(X_train_scaled, y_train)
                    best_model = gs.best_estimator_
                except Exception as e:
                    st.error(f"Error in model training ({model_name}): {e}")
                    continue

                # Predictions
                y_train_pred = best_model.predict(X_train_scaled)
                y_val_pred = best_model.predict(X_val_scaled)
                y_test_pred = best_model.predict(X_test_scaled)

                train_df = pd.DataFrame({'True': y_train, 'Predicted': y_train_pred})
                val_df = pd.DataFrame({'True': y_val, 'Predicted': y_val_pred})
                test_df = pd.DataFrame({'True': y_test, 'Predicted': y_test_pred})

                train_metrics = compute_metrics(y_train, y_train_pred)
                val_metrics = compute_metrics(y_val, y_val_pred)
                test_metrics = compute_metrics(y_test, y_test_pred)

                # --- FORECASTING ---
                preds_all_rows = []
                for idx, row in df_encoded.iterrows():
                    row_dict = {
                        'CrimeCategory': df.loc[idx, 'CrimeCategory'],
                        'ProvinceCode': df.loc[idx, 'ProvinceCode'],
                        'PoliceStationCode': df.loc[idx, 'PoliceStationCode'],
                        'Quarter': df.loc[idx, 'Quarter_Original']
                    }

                    if row[feature_cols].isnull().any():
                        continue

                    try:
                        feature_values = row[feature_cols].values.astype(float).reshape(1, -1)
                        scaler_row = MinMaxScaler()
                        feature_scaled = scaler_row.fit_transform(feature_values)

                        prev_years_scaled = feature_scaled.flatten().tolist()

                        for year in forecast_years:
                            input_vals = np.array(prev_years_scaled[-len(feature_cols):]).reshape(1, -1)
                            pred_scaled = best_model.predict(input_vals)[0]

                            prev_years_scaled.append(pred_scaled)
                            temp_all_years = np.array(prev_years_scaled[-len(feature_cols):]).reshape(1, -1)
                            pred_unscaled = scaler_row.inverse_transform(temp_all_years)[0, -1]
                            row_dict[f"{year}_Pred"] = pred_unscaled

                            if len(preds_all_rows) > 0:
                                prev_val = preds_all_rows[-1].get(f"{year-1}_Pred", np.nan)
                                if not pd.isna(prev_val):
                                    row_dict[f"Diff_{year}"] = ((prev_val - pred_unscaled) / prev_val) * 100
                                else:
                                    row_dict[f"Diff_{year}"] = np.nan
                            else:
                                row_dict[f"Diff_{year}"] = np.nan

                    except Exception as e:
                        print(f"Error forecasting row {idx}: {e}")
                        for year in forecast_years:
                            row_dict[f"{year}_Pred"] = np.nan
                            row_dict[f"Diff_{year}"] = np.nan

                    preds_all_rows.append(row_dict)

                preds_df = pd.DataFrame(preds_all_rows)

                if le_psc and pd.api.types.is_integer_dtype(preds_df['PoliceStationCode']):
                    preds_df['PoliceStationCode'] = le_psc.inverse_transform(preds_df['PoliceStationCode'])

                results[model_name] = {
                    'future_preds': preds_df,
                    'train_df': train_df,
                    'val_df': val_df,
                    'test_df': test_df,
                    'metrics': {
                        'Train': train_metrics,
                        'Validation': val_metrics,
                        'Test': test_metrics
                    }
                }

            return results


        # ---- MAIN EXECUTION ----
        df = df_fetch_all_stats_province_quarterly.copy()
        feature_cols = [str(y) for y in range(2016, 2023)]  # 2016–2022
        forecast_years = [2024, 2025, 2026]

        scenarios = {
            "Scenario 1": scenario_1,
            "Scenario 2": scenario_2,
            "Scenario 3": scenario_3,
            "Scenario 4": scenario_4,
            "Scenario 5": scenario_5
        }

        crime_categories = df['CrimeCategory'].unique()

        for label, func in scenarios.items():
            for cat in crime_categories:
                st.title(f"{label} - Recursive Forecasting with Evaluation: {cat}")
                df_cat = df[df['CrimeCategory'] == cat].reset_index(drop=True)

                if df_cat.shape[0] < 10:
                    st.warning(f"Not enough data for category: {cat}")
                    continue

                results = run_recursive_forecast(df_cat, feature_cols, forecast_years, func)

                for alg, res in results.items():
                    st.subheader(f"{alg} - Forecasts")
                    st.dataframe(res['future_preds'])

                    # === Metrics Bar Plot ===
                    st.markdown("### Evaluation Metrics (Bar Plot)")
                    metrics_df = pd.DataFrame(res['metrics']).T  # Transpose for bar plot
                    st.bar_chart(metrics_df)

                    # === Heatmaps for each forecast year ===
                    for year in forecast_years:
                        st.markdown(f"### Heatmap: {year} Forecasts per Police Station")
                        heat_data = res['future_preds'].pivot_table(
                            index='PoliceStationCode',
                            columns='Quarter',
                            values=f"{year}_Pred",
                            aggfunc='mean'
                        )
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.heatmap(heat_data, cmap='YlGnBu', annot=False, ax=ax)
                        ax.set_title(f"{year} Predictions Heatmap")
                        st.pyplot(fig)

                    # === Optional: View prediction breakdowns ===
                    st.markdown("### Train / Validation / Test Predictions")
                    prediction_col = st.columns(3)
                    with prediction_col[0]:
                        st.markdown("**Train**")
                        st.dataframe(res['train_df'])

                    with prediction_col[1]:
                        st.markdown("**Validation**")
                        st.dataframe(res['val_df'])

                    with prediction_col[2]:
                        st.markdown("**Test**")
                        st.dataframe(res['test_df'])

                    st.markdown("---")

        with ModelTrainingMerticsTab6:
                st.header(f'Model Training Metrics', divider='rainbow')
                for alg, res in results.items():
                    st.subheader(f"**{alg}**")

                    st.markdown("### Evaluation Metrics")
                    st.write("**Train**")
                    st.json(res['metrics']['Train'])
                    st.write("**Validation**")
                    st.json(res['metrics']['Validation'])
                    st.write("**Test**")
                    st.json(res['metrics']['Test'])

                    st.markdown("---")


with PostHocAnalysisTab7:
        st.header(f'Post Hoc Analysis - Interpretability', divider='rainbow')

       




