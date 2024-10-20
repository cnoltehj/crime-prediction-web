import requests
import pandas as pd
import config
from modelPredictionResponse.DataModel import PredictionData, MerticData



def fetch_prediction_province_policestation():

    endpoint = f"{config.BaseUrl_fetch_prediction_province_policestation}"
      
    try:
        response = requests.get(endpoint)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()  # Get the JSON data
        
        # Check if the data is in the expected format
        if isinstance(data, list):
            # Convert the JSON data to a DataFrame
            df = pd.DataFrame(data)
                    
            return df
        else:
            print("Unexpected data format received from the API")
            return pd.DataFrame()
    except requests.exceptions.HTTPError as errh:
        print(f"HTTP Error: {errh}")
    except requests.exceptions.ConnectionError as errc:
        print(f"Error Connecting: {errc}")
    except requests.exceptions.Timeout as errt:
        print(f"Timeout Error: {errt}")
    except requests.exceptions.RequestException as err:
        print(f"Error: {err}")
    
    return pd.DataFrame()  # Return an empty DataFrame if there's an error    

def fetch_all_provinces():
       
    endpoint = f"{config.BaseUrl_fetch_all_provinces}"
      
    try:
        response = requests.get(endpoint)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()  # Get the JSON data
        
        # Check if the data is in the expected format
        if isinstance(data, list):
            # Convert the JSON data to a DataFrame
            df = pd.DataFrame(data)
                    
            return df
        else:
            print("Unexpected data format received from the API")
            return pd.DataFrame()  # Return an empty DataFrame if the data format is unexpected
    
    except requests.exceptions.HTTPError as errh:
        print(f"HTTP Error: {errh}")
    except requests.exceptions.ConnectionError as errc:
        print(f"Error Connecting: {errc}")
    except requests.exceptions.Timeout as errt:
        print(f"Timeout Error: {errt}")
    except requests.exceptions.RequestException as err:
        print(f"Error: {err}")
    
    return pd.DataFrame()  # Return an empty DataFrame if there's an error

def fetch_policestation_per_provinces(provincecode: str):
       
    endpoint = f"{config.BaseUrl_fetch_policestation_per_provinces}provincecode={provincecode}"
      
    try:
        response = requests.get(endpoint)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()  # Get the JSON data
        
        # Check if the data is in the expected format
        if isinstance(data, list):
            # Convert the JSON data to a DataFrame
            df = pd.DataFrame(data)
                    
            return df
        else:
            print("Unexpected data format received from the API")
            return pd.DataFrame()  # Return an empty DataFrame if the data format is unexpected
    
    except requests.exceptions.HTTPError as errh:
        print(f"HTTP Error: {errh}")
    except requests.exceptions.ConnectionError as errc:
        print(f"Error Connecting: {errc}")
    except requests.exceptions.Timeout as errt:
        print(f"Timeout Error: {errt}")
    except requests.exceptions.RequestException as err:
        print(f"Error: {err}")
    
    return pd.DataFrame()  # Return an empty DataFrame if there's an error

def fetch_predition_province_policestation_year_quarterly_algorithm(provincecode: str, policestationcode: str, quarter: str,  algorithm: str):
       
    endpoint = f"{config.BaseUrl_fetch_predition_province_policestation_quarterly_algorithm}provincecode={provincecode}&policestationcode={policestationcode}&quarter={quarter}&algorithm={algorithm}"
      
    try:
        response = requests.get(endpoint)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()  # Get the JSON data
        
        # Print the API response for debugging purposes
        print("API response:", data)
        
        # Check if the data is in the expected format (list of dictionaries)
        if isinstance(data, list):
            # Convert the JSON data to a DataFrame
            df = pd.DataFrame(data)
            
             # Ensure 'CrimeCategory', 'ProvinceName', and 'StationName' are treated as strings and handle None values
            df['Algorithm'] = df['Algorithm'].astype(str)
            df['CrimeCategoryCode'] = df['CrimeCategoryCode'].astype(str)
            df['CrimeTypeName'] = df['CrimeTypeName'].astype(str)
            df['ProvinceCode'] = df['ProvinceCode'].astype(str)
            df['PoliceStationCode'] = df['PoliceStationCode'].astype(str)
            df['StationName'] = df['StationName'].astype(str)

            # Ensure numeric columns are converted to floats, except for specified columns
            for col in df.columns:
                if col not in ['Algorithm','CrimeCategoryCode', 'ProvinceCode', 'CrimeTypeName','ProvinceCode','PoliceStationCode','StationName']:  # Skip the specified columns
                    df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric, coercing errors to NaN

            return df
        else:
            print("Unexpected data format received from the API. Data is not a list.")
            return pd.DataFrame()  # Return an empty DataFrame if the data format is unexpected
    
    except requests.exceptions.HTTPError as errh:
        print(f"HTTP Error: {errh}")
    except requests.exceptions.ConnectionError as errc:
        print(f"Error Connecting: {errc}")
    except requests.exceptions.Timeout as errt:
        print(f"Timeout Error: {errt}")
    except requests.exceptions.RequestException as err:
        print(f"Error: {err}")
    
    return pd.DataFrame()  # Return an empty DataFrame if there's an error
 
def fetch_stats_province_policestation(provincecode: str, policestationcode: str, quarter: int):
       
    endpoint = f"{config.BaseUrl_fetch_stats_province_policestation}provincecode={provincecode}&policestationcode={policestationcode}"
      
    try:
        response = requests.get(endpoint)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()  # Get the JSON data
        
        # Check if the data is in the expected format
        if isinstance(data, list):
            # Convert the JSON data to a DataFrame
            df = pd.DataFrame(data)

            # Ensure numeric columns are converted to floats
            for col in df.columns:
                if col != 'CrimeCategory':  # Skip the CrimeCategory column
                    df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric, coercing errors to NaN
                    #print(df) 
                   
            return df
        else:
            print("Unexpected data format received from the API")
            return pd.DataFrame()  # Return an empty DataFrame if the data format is unexpected
    
    except requests.exceptions.HTTPError as errh:
        print(f"HTTP Error: {errh}")
    except requests.exceptions.ConnectionError as errc:
        print(f"Error Connecting: {errc}")
    except requests.exceptions.Timeout as errt:
        print(f"Timeout Error: {errt}")
    except requests.exceptions.RequestException as err:
        print(f"Error: {err}")
    
    return pd.DataFrame()  # Return an empty DataFrame if there's an error

def fetch_predition_province_policestation_year_quarterly_algorithm(provincecode: str, policestationcode: str, quarter: int, algorithm: str):
    endpoint = f"{config.BaseUrl_fetch_predition_province_policestation_quarterly_algorithm}provincecode={provincecode}&policestationcode={policestationcode}&quarter={quarter}&algorithm={algorithm}"
    
    try:
        response = requests.get(endpoint)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()  # Get the JSON data
        
        # Check if the data is in the expected format
        if isinstance(data, list) and len(data) > 0:
            # Convert the JSON data to a DataFrame
            df = pd.DataFrame(data)
            
            # Ensure specific columns are present and fill missing ones with 'N/A' or default values
            required_columns = ['CrimeCategory','Algorithm', 'ProvinceCode', 'PoliceStationCode', 'StationName']
            for col in required_columns:
                if col not in df.columns:
                    df[col] = 'N/A'  # Add the column and fill it with 'N/A'

            # Fill any missing values in these specific columns
            df['CrimeCategory'].fillna('N/A', inplace=True)
            df['Algorithm'].fillna('N/A', inplace=True)
            df['ProvinceCode'].fillna('N/A', inplace=True)
            df['PoliceStationCode'].fillna('N/A', inplace=True)
            df['StationName'].fillna('N/A', inplace=True)
            
            # Ensure numeric columns are handled
            for col in df.columns:
                if col not in required_columns:  # Skip non-numeric columns
                    df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric, coercing errors to NaN

            return df
        else:
            print("Unexpected data format or empty response from the API")
            return pd.DataFrame()  # Return an empty DataFrame if the data format is unexpected
    
    except requests.exceptions.HTTPError as errh:
        print(f"HTTP Error: {errh}")
    except requests.exceptions.ConnectionError as errc:
        print(f"Error Connecting: {errc}")
    except requests.exceptions.Timeout as errt:
        print(f"Timeout Error: {errt}")
    except requests.exceptions.RequestException as err:
        print(f"Error: {err}")
    
    return pd.DataFrame()  # Return an empty DataFrame if there's an error

def fetch_stats_province_quarterly(provincecode: str, quarter: int):
    endpoint = f"{config.BaseUrl_fetch_stats_province_quarterly}provincecode={provincecode}&quarter={quarter}"
    
    try:
        response = requests.get(endpoint)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()  # Get the JSON data
        
        # Check if the data is in the expected format
        if isinstance(data, list):
            # Convert the JSON data to a DataFrame
            df = pd.DataFrame(data)
            print(df)
            # Ensure numeric columns are converted to floats, except for specified columns
            for col in df.columns:
                if col not in ['CrimeCategory', 'ProvinceName', 'StationName']:  # Skip the specified columns
                    df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric, coercing errors to NaN
            print(df)
            return df
        else:
            print("Unexpected data format received from the API")
            return pd.DataFrame()  # Return an empty DataFrame if the data format is unexpected
    
    except requests.exceptions.HTTPError as errh:
        print(f"HTTP Error: {errh}")
    except requests.exceptions.ConnectionError as errc:
        print(f"Error Connecting: {errc}")
    except requests.exceptions.Timeout as errt:
        print(f"Timeout Error: {errt}")
    except requests.exceptions.RequestException as err:
        print(f"Error: {err}")
    
    return pd.DataFrame()  # Return an empty DataFrame if there's an error

def fetch_suggest_stats_province_policestation(provincecode: str, policestationcode: str):
       
    endpoint = f"{config.BaseUrl_fetch_suggest_stats_province_policestation}provincecode={provincecode}&policestationcode={policestationcode}"
      
    try:
        response = requests.get(endpoint)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()  # Get the JSON data
        
        # Check if the data is in the expected format
        if isinstance(data, list):
            # Convert the JSON data to a DataFrame
            df = pd.DataFrame(data)

            # Ensure numeric columns are converted to floats
            for col in df.columns:
                if col != 'CrimeCategory':  # Skip the CrimeCategory column
                    df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric, coercing errors to NaN
                    
            return df
        else:
            print("Unexpected data format received from the API")
            return pd.DataFrame()  # Return an empty DataFrame if the data format is unexpected
    
    except requests.exceptions.HTTPError as errh:
        print(f"HTTP Error: {errh}")
    except requests.exceptions.ConnectionError as errc:
        print(f"Error Connecting: {errc}")
    except requests.exceptions.Timeout as errt:
        print(f"Timeout Error: {errt}")
    except requests.exceptions.RequestException as err:
        print(f"Error: {err}")
    
    return pd.DataFrame()  # Return an empty DataFrame if there's an error

def fetch_stats_policestation_per_province(provincecode: str, quarter: int):
      
    endpoint = f"{config.BaseUrl_fetch_all_stats_policestation_per_province}provincecode={provincecode}&quarter={quarter}"
      
    try:
        response = requests.get(endpoint)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()  # Get the JSON data
        
        # Check if the data is in the expected format
        if isinstance(data, list):
            # Convert the JSON data to a DataFrame
            df = pd.DataFrame(data)
            
            # Ensure 'CrimeCategory', 'ProvinceName', and 'StationName' are treated as strings and handle None values
            df['CrimeCategory'] = df['CrimeCategory'].astype(str)
            df['ProvinceCode'] = df['ProvinceCode'].astype(str).fillna('Unknown ProvinceCode')
            df['PoliceStationCode'] = df['PoliceStationCode'].astype(str).fillna('Unknown PoliceStationCode')
            df['Quarter'] = df['Quarter'].astype(str).fillna('Unknown Quarter')

            # Ensure numeric columns are converted to floats, except for specified columns
            for col in df.columns:
                if col not in ['CrimeCategory', 'ProvinceCode', 'PoliceStationCode','Quarter']:  # Skip the specified columns
                    df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric, coercing errors to NaN

            return df
        else:
            print("Unexpected data format received from the API")
            return pd.DataFrame()  # Return an empty DataFrame if the data format is unexpected
    
    except requests.exceptions.HTTPError as errh:
        print(f"HTTP Error: {errh}")
    except requests.exceptions.ConnectionError as errc:
        print(f"Error Connecting: {errc}")
    except requests.exceptions.Timeout as errt:
        print(f"Timeout Error: {errt}")
    except requests.exceptions.RequestException as err:
        print(f"Error: {err}")
    
    return pd.DataFrame()

# Function to save prediction data via API

    endpoint = f"{config.BaseUrl_save_prediction_data}"  # Replace with your actual API endpoint

    for _, row in predictions_df.iterrows():
        scenario_number = int(row['Scenario'].split()[-1])  # Splits by space and takes the last part
        # Prepare JSON payload for each row in the DataFrame
        payload = PredictionData(
            Prediction=int(row['Prediction']),
            TrueValue=int(row['True_value']),
            Algorithm=row['Algorithm'],
            Scenario= scenario_number,
            CrimeCategoryCode=row['CrimeCategory'],
            ProvinceCode=row['ProvinceCode'],
            PoliceStationCode=row['PoliceStationCode'],
            Quarter=int(row['Quarter']),
            PredictionYear= 2024 #int(row['PredictionYear'])  
        ).model_dump()  # Convert Pydantic model to a dictionary

        try:
            response = requests.post(endpoint, json=payload)
            response.raise_for_status()  # Check for HTTP errors
            print(f"Prediction saved successfully for {row['Algorithm']} - {row['Scenario']}")
        except requests.exceptions.HTTPError as errh:
            print(f"HTTP Error: {errh}")
        except requests.exceptions.ConnectionError as errc:
            print(f"Error Connecting: {errc}")
        except requests.exceptions.Timeout as errt:
            print(f"Timeout Error: {errt}")
        except requests.exceptions.RequestException as err:
            print(f"Error: {err}")

#training_metrics_data
def fetch_training_metrics_data():
       
    endpoint = f"{config.BaseUrl_fetch_training_metrics_data}"
      
    try:
        response = requests.get(endpoint)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()  # Get the JSON data
        
        # Check if the data is in the expected format
        if isinstance(data, list):
            # Convert the JSON data to a DataFrame
            df = pd.DataFrame(data)
                    
            return df
        else:
            print("Unexpected data format received from the API")
            return pd.DataFrame()  # Return an empty DataFrame if the data format is unexpected
    
    except requests.exceptions.HTTPError as errh:
        print(f"HTTP Error: {errh}")
    except requests.exceptions.ConnectionError as errc:
        print(f"Error Connecting: {errc}")
    except requests.exceptions.Timeout as errt:
        print(f"Timeout Error: {errt}")
    except requests.exceptions.RequestException as err:
        print(f"Error: {err}")
    
    return pd.DataFrame()  # Return an empty DataFrame if there's an error

# Function to save prediction data via API
def save_train_prediction_data(predictions_df):
    # Remove the trailing `/?` if unnecessary
    endpoint = config.BaseUrl_save_trained_prediction_data

    for _, row in predictions_df.iterrows():
        # Handle scenario extraction from string (e.g., 'Scenario 1' -> 1)
        try:
            scenario_number = int(row['Scenario'].split()[-1])  # Assumes 'Scenario X' format
        except ValueError:
            print(f"Error parsing scenario for row: {row}")
            continue  # Skip this row if scenario parsing fails

        # Prepare JSON payload using Pydantic model
        payload = PredictionData(
            Prediction=int(row['Prediction']),
            TrueValue=int(row['True_value']),
            Algorithm=row['Algorithm'],
            Scenario=scenario_number,
            CrimeCategoryCode=row['CrimeCategory'],
            ProvinceCode=row['ProvinceCode'],
            PoliceStationCode=row['PoliceStationCode'],
            Quarter=int(row['Quarter']),
            PredictionYear=2024  # Static value for prediction year
        ).model_dump()  # Correctly invoke model_dump()

        # Debug: Print payload for inspection before sending
        print(f"Sending payload: {payload}")

        try:
            # Send POST request with the prepared payload
            response = requests.post(endpoint, json=payload)
            response.raise_for_status()  # Raise exception for HTTP errors
            print(f"Prediction saved successfully for {row['Algorithm']} - Scenario {scenario_number}")

        except requests.exceptions.HTTPError as errh:
            print(f"HTTP Error: {errh}")
        except requests.exceptions.ConnectionError as errc:
            print(f"Error Connecting: {errc}")
        except requests.exceptions.Timeout as errt:
            print(f"Timeout Error: {errt}")
        except requests.exceptions.RequestException as err:
            print(f"Error: {err}")

def save_all_prediction_data(predictions_df):
    # Remove the trailing `/?` if unnecessary
    endpoint = config.BaseUrl_save_all_prediction_data

    for _, row in predictions_df.iterrows():
        # Handle scenario extraction from string (e.g., 'Scenario 1' -> 1)
        try:
            scenario_number = int(row['Scenario'].split()[-1])  # Assumes 'Scenario X' format
        except ValueError:
            print(f"Error parsing scenario for row: {row}")
            continue  # Skip this row if scenario parsing fails

        # Prepare JSON payload using Pydantic model
        payload = PredictionData(
            Prediction=int(row['Prediction']),
            TrueValue=int(row['True_value']),
            Algorithm=row['Algorithm'],
            Scenario=scenario_number,
            CrimeCategoryCode=row['CrimeCategory'],
            ProvinceCode=row['ProvinceCode'],
            PoliceStationCode=row['PoliceStationCode'],
            Quarter=int(row['Quarter']),
            PredictionYear=2024  # Static value for prediction year
        ).model_dump()  # Correctly invoke model_dump()

        # Debug: Print payload for inspection before sending
        print(f"Sending payload: {payload}")

        try:
            # Send POST request with the prepared payload
            response = requests.post(endpoint, json=payload)
            response.raise_for_status()  # Raise exception for HTTP errors
            print(f"Prediction saved successfully for {row['Algorithm']} - Scenario {scenario_number}")

        except requests.exceptions.HTTPError as errh:
            print(f"HTTP Error: {errh}")
        except requests.exceptions.ConnectionError as errc:
            print(f"Error Connecting: {errc}")
        except requests.exceptions.Timeout as errt:
            print(f"Timeout Error: {errt}")
        except requests.exceptions.RequestException as err:
            print(f"Error: {err}")

def save_metric_data(metrics_df):
    endpoint = f"{config.BaseUrl_save_mertic_data}"  # Replace with your actual API endpoint

    for _, row in metrics_df.iterrows():

        scenario_number = int(row['Scenario'].split()[-1])
        # Prepare JSON payload for each row in the DataFrame
        payload = MerticData(
        Algorithm = row['Algorithm'], 
        Scenario = scenario_number,
        PredictedYear = 2024, #int(row['PredictedYear']),
        MAE = float(row['MAE']),
        MSE = float(row['MSE']),
        MAPE = float(row['MAPE']),
        RSquare = float(row['R²']),
        ARS = float(row['ARS']) 
        ).model_dump()  # Convert Pydantic model to a dictionary

        try:
            response = requests.post(endpoint, json=payload)
            response.raise_for_status()  # Check for HTTP errors
            print(f"Prediction saved successfully for {row['Algorithm']} - {row['Scenario']}")
        except requests.exceptions.HTTPError as errh:
            print(f"HTTP Error: {errh}")
        except requests.exceptions.ConnectionError as errc:
            print(f"Error Connecting: {errc}")
        except requests.exceptions.Timeout as errt:
            print(f"Timeout Error: {errt}")
        except requests.exceptions.RequestException as err:
            print(f"Error: {err}")