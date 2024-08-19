import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer


# Initialize encoders
#label_encoder = LabelEncoder()
one_hot_encoder = OneHotEncoder(sparse_output=False)

# Scenarios Encoding ColumnTransformer
def get_transformer_encode(scenario):
    if scenario == 1:
        return ColumnTransformer([
            ('PoliceStationCode', 'passthrough', ['PoliceStationCode']),
            ('Quarter', 'passthrough', ['Quarter'])
        ], remainder='passthrough')
    elif scenario == 2:
        return ColumnTransformer([
            ('PoliceStationCode', one_hot_encoder, ['PoliceStationCode']),
            ('Quarter', one_hot_encoder, ['Quarter'])
        ], remainder='passthrough')
    elif scenario == 3:
        return ColumnTransformer([
            ('PoliceStationCode', 'passthrough', ['PoliceStationCode']),
            ('Quarter', one_hot_encoder, ['Quarter'])
        ], remainder='passthrough')
    elif scenario == 4:
        return ColumnTransformer([
            ('PoliceStationCode', one_hot_encoder, ['PoliceStationCode']),
            ('Quarter', 'passthrough', ['Quarter'])
        ], remainder='passthrough')

# Scenario 1: Label encoding for 'PoliceStationCode' and 'Quarter'
def scenario_1(df):
    label_encoder = LabelEncoder()
    df['PoliceStationCode'] = label_encoder.fit_transform(df['PoliceStationCode'])
    df['Quarter'] = label_encoder.fit_transform(df['Quarter'])
    return df

# Scenario 2: One-hot encoding for 'PoliceStationCode' and 'Quarter'
def scenario_2(df):
    onehot_encoder = ColumnTransformer(
        transformers=[
            ('police', OneHotEncoder(), ['PoliceStationCode']),
            ('quarter', OneHotEncoder(), ['Quarter'])
        ],
        remainder='passthrough'
    )
    return pd.DataFrame(onehot_encoder.fit_transform(df))

# Scenario 3: Label encoding for 'PoliceStationCode' and one-hot encoding for 'Quarter'
def scenario_3(df):
    label_encoder = LabelEncoder()
    df['PoliceStationCode'] = label_encoder.fit_transform(df['PoliceStationCode'])
    
    onehot_encoder = OneHotEncoder()
    quarter_encoded = onehot_encoder.fit_transform(df[['Quarter']])
    df = pd.concat([df.drop(columns=['Quarter']), pd.DataFrame(quarter_encoded.toarray())], axis=1)
    return df

# Scenario 4: Label encoding for 'Quarter' and one-hot encoding for 'PoliceStationCode'
def scenario_4(df):
    label_encoder = LabelEncoder()
    df['Quarter'] = label_encoder.fit_transform(df['Quarter'])
    
    onehot_encoder = OneHotEncoder()
    police_encoded = onehot_encoder.fit_transform(df[['PoliceStationCode']])
    df = pd.concat([df.drop(columns=['PoliceStationCode']), pd.DataFrame(police_encoded.toarray())], axis=1)
    return df


def apply_encodings(X, label_cols):
    transformers = []

    for col in label_cols:
        transformers.append((f'label_{col}', 
                             FunctionTransformer(lambda df, col=col: pd.DataFrame(LabelEncoder().fit_transform(df[col]), columns=[col]), 
                                                 validate=False), 
                             [col]))

    preprocessor = ColumnTransformer(transformers, remainder='passthrough')
    X_transformed = preprocessor.fit_transform(X)

    # Create a DataFrame with the correct column names
    # The first few columns are the encoded ones, followed by the passthrough columns
    encoded_columns = label_cols
    passthrough_columns = [col for col in X.columns if col not in label_cols]
    new_column_order = encoded_columns + passthrough_columns
    X_transformed_df = pd.DataFrame(X_transformed, columns=new_column_order)

    return X_transformed_df, preprocessor

def label_encode_features(df, columns):
    le = {}
    df_encoded = df.copy()
    for col in columns:
        le[col] = LabelEncoder()
        df_encoded[col] = le[col].fit_transform(df_encoded[col])
    return df_encoded

def one_hot_encode_features(df, columns):
    return pd.get_dummies(df, columns=columns)


def label_encode_features_original(df):
   # Use Label Encoding or One-Hot Encoding on categorical features
    categorical_cols = ['CrimeCategory', 'ProvinceCode', 'PoliceStationCode', 'Quarter']
    
    # Option 1: Label Encoding
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    return df

def one_hot_encode_features_original(df):
    categorical_cols = ['CrimeCategory', 'ProvinceCode', 'PoliceStationCode', 'Quarter']
    df = pd.get_dummies(df, columns=categorical_cols)
    return df
    #return pd.get_dummies(df, columns=columns)

