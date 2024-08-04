{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "         0      1     2     3      4     5      6\n",
      "Row1  27.3  18.71  -3.7 -57.7  27.30  21.4  23.53\n",
      "Row2  20.4 -18.20 -40.9 -57.0   6.35  -7.6 -25.51\n",
      "Row3  -2.6  36.90 -29.6 -84.5  51.37  -7.8 -26.05\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import math\n",
    "\n",
    "# Define the data rows\n",
    "data_rows = {\n",
    "    'Row1': [27.3, 92.9, -3.7, -57.7, 27.3, 21.4, 23.53],\n",
    "    'Row2': [20.4,\t-18.2,\t-40.9,\t-57.0,\t173.3,\t-7.6, -25.51],\n",
    "    'Row3': [-2.6, 36.9, -29.6, -84.5, 473.3, -7.8, -26.05]\n",
    "}\n",
    "\n",
    "# Create a DataFrame\n",
    "df = pd.DataFrame(data_rows).T  # Transpose to have rows as DataFrame rows\n",
    "\n",
    "# Define the outliers and their replacement logic\n",
    "outliers = {\n",
    "    'Row1': 92.9,\n",
    "    'Row2': 173.3,\n",
    "    'Row3': 473.3\n",
    "}\n",
    "\n",
    "# Function to round down to 2 decimal places\n",
    "def round_down(value, decimals=2):\n",
    "    factor = 10.0 ** decimals\n",
    "    return math.floor(value * factor) / factor\n",
    "\n",
    "# Replace outliers with the rounded down median value\n",
    "for row, outlier in outliers.items():\n",
    "    # Calculate the median of the row\n",
    "    mean_value = np.mean(df.loc[row])\n",
    "    \n",
    "    # Round down to 2 decimal places\n",
    "    mean_value_rounded = round_down(mean_value, 2)\n",
    "    \n",
    "    # Replace outlier with rounded median value\n",
    "    df.loc[row] = df.loc[row].replace(outlier, mean_value_rounded)\n",
    "\n",
    "\n",
    "# Display the updated DataFrame\n",
    "print(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'dataRequest'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[10], line 1\u001b[0m\n\u001b[1;32m----> 1\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mdataRequest\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mcrimedbRequest\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m (\n\u001b[0;32m      2\u001b[0m     fetch_stats_policestation_per_province\n\u001b[0;32m      3\u001b[0m     )\n\u001b[0;32m      5\u001b[0m df_suggeted_province_quarterly_data_db \u001b[38;5;241m=\u001b[39m fetch_stats_policestation_per_province(\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mZA.WC\u001b[39m\u001b[38;5;124m'\u001b[39m,\u001b[38;5;241m1\u001b[39m)\n",
      "\u001b[1;31mModuleNotFoundError\u001b[0m: No module named 'dataRequest'"
     ]
    }
   ],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "\n",
    "def identify_outliers_data(outliers: pd.DataFrame):\n",
    "\n",
    "    try:\n",
    "        df = pd.DataFrame(outliers)\n",
    "\n",
    "        # Check if the data is in the expected format\n",
    "        if not df.empty:\n",
    "            # Convert 'Percentage' columns to numeric\n",
    "            percentage_cols = [col for col in df.columns if col not in ['CrimeCategory', 'ProvinceCode', 'PoliceStationCode', 'Quarter']]\n",
    "            df[percentage_cols] = df[percentage_cols].apply(pd.to_numeric, errors='coerce')\n",
    "\n",
    "            # Identify outliers per row\n",
    "            outliers_list = []\n",
    "\n",
    "            for index, row in df.iterrows():\n",
    "                row_data = row[percentage_cols]\n",
    "                Q1 = row_data.quantile(0.25)\n",
    "                Q3 = row_data.quantile(0.75)\n",
    "                IQR = Q3 - Q1\n",
    "                lower_bound = Q1 - 1.5 * IQR\n",
    "                upper_bound = Q3 + 1.5 * IQR\n",
    "\n",
    "                row_outliers = row_data[(row_data < lower_bound) | (row_data > upper_bound)]\n",
    "                \n",
    "                if not row_outliers.empty:\n",
    "                    outlier_row = row.copy()\n",
    "                    outlier_row['Outliers'] = row_outliers.values\n",
    "                    outliers_list.append(outlier_row)\n",
    "\n",
    "            outliers_df = pd.DataFrame(outliers_list)\n",
    "\n",
    "            return outliers_df\n",
    "\n",
    "        else:\n",
    "            print(\"Unexpected data format received from the API\")\n",
    "            return pd.DataFrame()  # Return an empty DataFrame if the data format is unexpected\n",
    "\n",
    "    except Exception as e:\n",
    "        print(f\"Error: {e}\") # Return an empty DataFrame if the data format is unexpected\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "\n",
    "def replace_outliers_data(outliers: pd.DataFrame):\n",
    "    df = pd.DataFrame(outliers)\n",
    "\n",
    "    # Check if the data is in the expected format\n",
    "    if not df.empty:\n",
    "        # Function to calculate IQR and replace outliers with the row mean\n",
    "        def replace_outliers(row):\n",
    "            Q1 = row.quantile(0.25)\n",
    "            Q3 = row.quantile(0.75)\n",
    "            IQR = Q3 - Q1\n",
    "            outlier_mask = (row < (Q1 - 1.5 * IQR)) | (row > (Q3 + 1.5 * IQR))\n",
    "            mean_value = row[~outlier_mask].mean().round(2)\n",
    "            row[outlier_mask] = mean_value\n",
    "            return row\n",
    "\n",
    "        # Select only the percentage columns (which are the last 7 columns)\n",
    "        percentage_columns = df.columns[4:]\n",
    "\n",
    "        # Apply the function to each row\n",
    "        df[percentage_columns] = df[percentage_columns].apply(replace_outliers, axis=1)\n",
    "\n",
    "        return df\n",
    "\n",
    "    else:\n",
    "        print(\"Unexpected data format received from the API\")\n",
    "        return pd.DataFrame()\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
