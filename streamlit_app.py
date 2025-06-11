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

        years = list(range(2016, 2024))  # 2016 through 2023

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

        # 2. Helper for metrics
        def compute_metrics(y_true, y_pred):
            return {
                'MAE': mean_absolute_error(y_true, y_pred),
                'MSE': mean_squared_error(y_true, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
                'R2': r2_score(y_true, y_pred),
                'MAPE': mean_absolute_percentage_error(y_true, y_pred)
            }

        # 3. Main routine
        def run_all(df, feature_cols, target_col, scenario_func):
            # --- Step 1: Save original fields before encoding ---
            original_fields = df[['CrimeCategory', 'ProvinceCode', 'PoliceStationCode', 'Quarter']].copy()

            # --- Step 2: Apply encoding scenario ---
            df_encoded = scenario_func(df.copy())  # apply scenario
            encoded_feature_cols = [col for col in df_encoded.columns if col in feature_cols]

            # --- Step 3: Split dataset ---
            X = df_encoded[encoded_feature_cols].values
            y = df_encoded[target_col].values
            meta = original_fields.copy()

            X_train, X_temp, y_train, y_temp, meta_train, meta_temp = train_test_split(X, y, meta, test_size=0.4, random_state=42)
            X_val, X_test, y_val, y_test, meta_val, meta_test = train_test_split(X_temp, y_temp, meta_temp, test_size=0.5, random_state=42)

            # --- Step 4: Scale (MinMax then Standard) ---
            minmax_scaler = MinMaxScaler()
            X_train = minmax_scaler.fit_transform(X_train)
            X_val = minmax_scaler.transform(X_val)
            X_test = minmax_scaler.transform(X_test)

            std_scaler = StandardScaler()
            X_train = std_scaler.fit_transform(X_train)
            X_val = std_scaler.transform(X_val)
            X_test = std_scaler.transform(X_test)

            # --- Step 5: Cross-validation ---
            cv = KFold(n_splits=3, shuffle=True, random_state=42)

            # --- Step 6: Grid Search & Prediction ---
            all_results = {}
            for name, model in models.items():
                gs = GridSearchCV(model, params[name], cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
                gs.fit(X_train, y_train)
                best = gs.best_estimator_

                # Predictions
                y_tr_pred = best.predict(X_train)
                y_val_pred = best.predict(X_val)
                y_te_pred = best.predict(X_test)

                # --- Step 7: Store with original categorical fields ---
                train_df = meta_train.copy()
                train_df['True'] = y_train
                train_df['Pred'] = y_tr_pred

                val_df = meta_val.copy()
                val_df['True'] = y_val
                val_df['Pred'] = y_val_pred

                test_df = meta_test.copy()
                test_df['True'] = y_test
                test_df['Pred'] = y_te_pred

                # --- Step 8: Store metrics and results ---
                all_results[name] = {
                    'train_pred': train_df,
                    'val_pred': val_df,
                    'test_pred': test_df,
                    'train_metrics': compute_metrics(y_train, y_tr_pred),
                    'val_metrics': compute_metrics(y_val, y_val_pred),
                    'test_metrics': compute_metrics(y_test, y_te_pred)
                }

            return all_results


        # 4. Usage & display in Streamlit
        df = df_fetch_all_stats_province_quarterly
        feature_cols = [str(y) for y in range(2016, 2023)]
        target_col = '2023'

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
            label_encoder_psc = LabelEncoder()
            label_encoder_qtr = LabelEncoder()
            df['PoliceStationCode'] = label_encoder_psc.fit_transform(df['PoliceStationCode'])
            df['Quarter'] = label_encoder_qtr.fit_transform(df['Quarter'])
            return df

        # --- 2. Now you can define the scenario dictionary safely ---
        scenarios = {
            "Scenario 1": scenario_1,
            "Scenario 2": scenario_2,
            "Scenario 3": scenario_3,
            "Scenario 4": scenario_4,
            "Scenario 5": scenario_5
            }

        for label, func in scenarios.items():
            st.title(f"{label} - Label & OneHot Encoding")
            results = run_all(df, feature_cols, target_col, func)

            for alg, res in results.items():
                st.subheader(f"**{alg}**")
                performance_col = st.columns((2, 2, 2))
                with performance_col[0]:
                    st.markdown("**Train Predictions**")
                    st.dataframe(res['train_pred'])
                with performance_col[1]:
                    st.markdown("**Validation Predictions**")
                    st.dataframe(res['val_pred'])
                with performance_col[2]:
                    st.markdown("**Test Predictions**")
                    st.dataframe(res['test_pred'])

                st.markdown("**Metrics**")
                st.json({
                    "Train": res['train_metrics'],
                    "Validation": res['val_metrics'],
                    "Test": res['test_metrics']
                })

                st.markdown("---")


with ModelTrainingValidationTab5:
        st.header(f'Model Training Validation', divider='rainbow')

        for alg, res in results.items():
            st.subheader(f"**{alg}**")

            st.markdown("**Validation Metrics**")
            st.json(res['val_metrics'])

            st.markdown("---")


with ModelTrainingMerticsTab6:
        st.header(f'Model Training Metrics', divider='rainbow')
        for alg, res in results.items():
            st.subheader(f"**{alg}**")

            st.markdown("**Test Metrics**")
            st.json(res['test_metrics'])

            st.markdown("---")


with PostHocAnalysisTab7:
        st.header(f'Post Hoc Analysis - Interpretability', divider='rainbow')

       




