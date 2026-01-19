import ast
import os
import pickle

import pandas as pd
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

load_dotenv()
MODEL_FILENAME = os.getenv("MODEL_FILENAME")
if not MODEL_FILENAME:
    raise ValueError("Set MODEL_FILENAME to .env")

DATASET_FILENAME = os.getenv("DATASET_FILENAME")
if not DATASET_FILENAME:
    raise ValueError("Set DATASET_FILENAME to .env")


categorical_columns = [
    "neighbourhood",
    "neighbourhood_cleansed",
    "neighbourhood_group_cleansed",
    "property_type",
    "room_type",
]
numerical_columns = [
    "host_response_rate",
    "host_acceptance_rate",
    "host_is_superhost",
    "host_has_profile_pic",
    "host_identity_verified",
    "latitude",
    "longitude",
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
    "minimum_nights",
    "maximum_nights",
    "is_licensed",
]


def train(df_train, y_train):
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.8,
        max_features=500,
        lowercase=True,
        stop_words="english",
    )
    tfidf.fit_transform(df_train["amenities_text"])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_columns),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
            ("amenities", tfidf, "amenities_text"),
        ]
    )
    X_train = preprocessor.fit_transform(df_train)
    dtr_model = DecisionTreeRegressor(
        max_depth=6,
        min_samples_leaf=20,
    )
    dtr_model.fit(X_train, y_train)

    return preprocessor, dtr_model

def clean_data(df) -> pd.DataFrame:
    # Converting Amenities to text and vectorizing results
    def amenities_to_text(amenities):
        return " ".join(amenities) if amenities else ""
    df["amenities"] = df["amenities"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    df["amenities_text"] = df["amenities"].apply(amenities_to_text)

    # Type casting truth values to integer
    columns_with_truth_values = [
        "host_is_superhost",
        "host_has_profile_pic",
        "host_identity_verified",
    ]
    for column in columns_with_truth_values:
        df[column] = (df[column] == "t").astype(int)

    # Adding a new column to understand whether a listing has a license to run or not
    df["is_licensed"] = (df.license.isnull() == False).astype(int)  # noqa: E712

    # Converting percentage numericals to numberical columns
    percentage_columns = [
        "host_response_rate",
        "host_acceptance_rate",
    ]
    for column in percentage_columns:
        df[column] = (
            df[column].astype(str).str.strip("%").replace("nan", pd.NA).astype("Int64")
        )
    
    # Converting price to float
    df["price"] = pd.to_numeric(
        df["price"].str.replace("€", "", regex=False), errors="coerce"
    )
    df = df.loc[df["price"].notna()].copy()

    return df


df = pd.read_csv(DATASET_FILENAME)

df = clean_data(df)

final_df = df[categorical_columns + numerical_columns + ["amenities_text", "price"]]

# Fill categorical NaNs
final_df.loc[:, categorical_columns] = final_df[categorical_columns].fillna("")

# Drop rows with missing numeric values
final_df = final_df.loc[~final_df[numerical_columns].isna().any(axis=1)].copy()

preprocessor, model = train(final_df, final_df.price.values)

with open(MODEL_FILENAME, "wb") as f_out:
    pickle.dump((preprocessor, model), f_out)

print("The model is saved to model.bin")