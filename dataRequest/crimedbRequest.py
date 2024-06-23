import requests
import pandas as pd
import config

def fetch_crime_data(provincecode: str, policestationcode: str, year: int, quarter: int):
       
    endpoint = f"{config.BaseUrl_crimestats}provincecode={provincecode}&policestation={policestationcode}&year={year}&quarter={quarter}"
      
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


def fetch_provinces_data():
       
    endpoint = f"{config.BaseUrl_allprovinces}"
      
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

def fetch_policestation_data(provincecode: str):
       
    endpoint = f"{config.BaseUrl_policestationperprovinces}provincecode={provincecode}"
      
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
