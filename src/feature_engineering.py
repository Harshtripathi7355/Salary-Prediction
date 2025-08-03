import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Drop missing
    df = df.dropna()

    # One-hot encode categorical
    cat_cols = df.select_dtypes(include="object").columns
    encoder = OneHotEncoder(sparse=False)
    encoded = encoder.fit_transform(df[cat_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))

    # Scale numeric
    numeric = df.select_dtypes(include=["int64", "float64"])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric)
    scaled_df = pd.DataFrame(scaled, columns=numeric.columns)

    return pd.concat([scaled_df, encoded_df], axis=1)
