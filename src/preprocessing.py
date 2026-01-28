import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(path):
    df = pd.read_csv(path)
    df_cleaned = df.copy()

    # Binary encoding
    binary_map = {'Yes': 1, 'No': 0}
    df_cleaned['is_male'] = df['gender'].map({'Male': 1, 'Female': 0})
    df_cleaned['is_Partner'] = df['Partner'].map(binary_map)
    df_cleaned['is_Dependents'] = df['Dependents'].map(binary_map)
    df_cleaned['is_PhoneService'] = df['PhoneService'].map(binary_map)
    df_cleaned['is_PaperlessBilling'] = df['PaperlessBilling'].map(binary_map)
    df_cleaned['is_Churn'] = df['Churn'].map(binary_map)

    # Handle TotalCharges
    df_cleaned['TotalCharges'] = pd.to_numeric(
        df_cleaned['TotalCharges'], errors='coerce'
    )
    df_cleaned['TotalCharges'].fillna(
        df_cleaned['TotalCharges'].median(), inplace=True
    )

    # Scale numeric columns
    scaler = StandardScaler()
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df_cleaned[num_cols] = scaler.fit_transform(df_cleaned[num_cols])

    # One-hot encoding
    df_cleaned = pd.get_dummies(
        df_cleaned,
        columns=[
            'PaymentMethod','Contract','MultipleLines','InternetService',
            'OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport',
            'StreamingTV','StreamingMovies'
        ],
        drop_first=True
    )

    X = df_cleaned.select_dtypes(include=['int64', 'float64']).drop('is_Churn', axis=1)
    y = df_cleaned['is_Churn']

    return X, y, df_cleaned, scaler
