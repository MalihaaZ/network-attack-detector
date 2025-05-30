import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def load_and_preprocess(parquet_path):
    df = pd.read_parquet(parquet_path)

    # Drop unnecessary columns if present
    df.drop(columns=['id', 'label_name'], errors='ignore', inplace=True)

    # Encode all object (categorical) columns
    label_encoders = {}
    for column in df.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le

    # Separate features and label
    X = df.drop('label', axis=1)
    y = df['label']

    # Scale the data
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X_scaled, y, scaler, label_encoders
