import requests
import pandas as pd
import config



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

def fetch_stats_province_policestation_quarterly(provincecode: str, policestationcode: str, quarter: int):
       
    endpoint = f"{config.BaseUrl_fetch_stats_province_policestation_quarterly}provincecode={provincecode}&policestationcode={policestationcode}&quarter={quarter}"
      
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