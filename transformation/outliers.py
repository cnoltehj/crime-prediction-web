import pandas as pd

def identify_outliers(outliers : pd.DataFrame):

    # try:

        df = pd.DataFrame(outliers)

        # Check if the data is in the expected format
        if not df.empty:

           # Melt the DataFrame to long format for easier plotting
            df_melted = df.melt(id_vars=["CrimeCategory"], var_name="Year", value_name="Percentage")

            # Convert 'Percentage' column to numeric
            df_melted['Percentage'] = pd.to_numeric(df_melted['Percentage'], errors='coerce')

            # Drop rows with NaN values in 'Percentage' column
            df_melted.dropna(subset=['Percentage'], inplace=True)

            # Calculate IQR and identify outliers
            Q1 = df_melted['Percentage'].quantile(0.25)
            Q3 = df_melted['Percentage'].quantile(0.75)
            IQR = Q3 - Q1

            # Define outliers
            outliers = df_melted[(df_melted['Percentage'] < (Q1 - 1.5 * IQR)) | (df_melted['Percentage'] > (Q3 + 1.5 * IQR))]
            
            return outliers

        else:
            print("Unexpected data format received from the API")
            return pd.DataFrame()  # Return an empty DataFrame if the data format is unexpected

        #  except:
        #     print("Unexpected data format received from the API")
        #     return pd.DataFrame()  # Return an empty DataFrame if the data format is unexpected


def replace_outliers(outliers : pd.DataFrame):

    # try:

        df = pd.DataFrame(outliers)

        # Check if the data is in the expected format
        if not df.empty:

            # Melt the DataFrame to long format for easier plotting
            df_melted = df.melt(id_vars=["CrimeCategory"], var_name="Year", value_name="Percentage")

            # Convert 'Percentage' column to numeric
            df_melted['Percentage'] = pd.to_numeric(df_melted['Percentage'], errors='coerce')

            # Drop rows with NaN values in 'Percentage' column
            df_melted.dropna(subset=['Percentage'], inplace=True)

            # Calculate IQR and identify outliers
            Q1 = df_melted['Percentage'].quantile(0.25)
            Q3 = df_melted['Percentage'].quantile(0.75)
            IQR = Q3 - Q1

            # Define outliers
            outliers = df_melted[(df_melted['Percentage'] < (Q1 - 1.5 * IQR)) | (df_melted['Percentage'] > (Q3 + 1.5 * IQR))]

            # Replace outliers with the median value of their respective Year
            median_values = df_melted.groupby('Year')['Percentage'].median()

            # Function to replace outliers with median
            def replace_outliers(row):
                if row.name in outliers.index:
                    return median_values[row['Year']]
                else:
                    return row['Percentage']

            # Apply the function
            df_melted['Percentage'] = df_melted.apply(replace_outliers, axis=1)

            # Pivot the DataFrame back to wide format for regression
            cleaned_df = df_melted.pivot(index="CrimeCategory", columns="Year", values="Percentage").reset_index() 

            return cleaned_df

        else:
            print("Unexpected data format received from the API")
            return pd.DataFrame()  # Return an empty DataFrame if the data format is unexpected