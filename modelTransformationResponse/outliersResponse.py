import pandas as pd
import math

def identify_outliers_data(outliers: pd.DataFrame):

    try:
        df = pd.DataFrame(outliers)

        # Check if the data is in the expected format
        if not df.empty:
            # Convert 'Percentage' columns to numeric
            percentage_cols = [col for col in df.columns if col not in ['CrimeCategory', 'ProvinceCode', 'PoliceStationCode', 'Quarter']]
            df[percentage_cols] = df[percentage_cols].apply(pd.to_numeric, errors='coerce')

            # Identify outliers per row
            outliers_list = []

            for index, row in df.iterrows():
                row_data = row[percentage_cols]
                Q1 = row_data.quantile(0.25)
                Q3 = row_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                row_outliers = row_data[(row_data < lower_bound) | (row_data > upper_bound)]
                
                if not row_outliers.empty:
                    outlier_row = row.copy()
                    outlier_row['Outliers'] = row_outliers.values
                    outliers_list.append(outlier_row)

            outliers_df = pd.DataFrame(outliers_list)

            return outliers_df

        else:
            print("Unexpected data format received from the API")
            return pd.DataFrame()  # Return an empty DataFrame if the data format is unexpected

    except Exception as e:
        print(f"Error: {e}") # Return an empty DataFrame if the data format is unexpected

def replace_outliers_data(outliers: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(outliers)

    if not df.empty:
        # Select only the percentage columns (which are the last 7 columns)
        percentage_columns = df.columns[4:]  # Adjust this if your columns are different

        def replace_outliers(row):
            # Ensure we're working with a Series object
            percentages = row[percentage_columns]
            if isinstance(percentages, pd.Series):
                Q1 = percentages.quantile(0.25)
                Q3 = percentages.quantile(0.75)
                IQR = Q3 - Q1
                
                # Create a boolean mask for outliers
                outlier_mask = (percentages < (Q1 - 1.5 * IQR)) | (percentages > (Q3 + 1.5 * IQR))
                
                # Calculate the mean of non-outlier values
                mean_value = percentages[~outlier_mask].mean()
                
                # Replace outliers with the mean value
                row[percentage_columns] = percentages.where(~outlier_mask, mean_value)
            return row

        # Apply the function to each row
        df = df.apply(replace_outliers, axis=1)
        
        return df
    else:
        print("Unexpected data format received from the API")
        return pd.DataFrame()
    
# Function to round down to 2 decimal places
def round_down(value, decimals=2):
    factor = 10.0 ** decimals
    return math.floor(value * factor) / factor