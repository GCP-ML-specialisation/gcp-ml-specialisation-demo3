from kfp.v2.dsl import component, Input, Output, Dataset


@component(
    packages_to_install=["pandas", "gcsfs", "scikit-learn", "numpy"],
    output_component_file="feature_engineering.yaml",
    base_image="python:3.9",
)
def data_transformation(
    df_train: Input[Dataset],
    df_test: Input[Dataset],
    dataset_train: Output[Dataset],
    dataset_test: Output[Dataset],
):

    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    import numpy as np

    df_train = pd.read_csv(df_train.path + ".csv")
    df_test = pd.read_csv(df_test.path + ".csv")

    # Handle categorical to integer transformation for 'Gender'
    gender_mapping = {"F": 0, "M": 1}
    df_train["Gender"] = df_train["Gender"].map(gender_mapping)
    df_test["Gender"] = df_test["Gender"].map(gender_mapping)

    # Columns to encode
    cols = ["Age", "City_Category", "Stay_In_Current_City_Years"]

    # Combine train and test for consistent encoding
    combined_df = pd.concat([df_train[cols], df_test[cols]], axis=0)

    # Initialize the LabelEncoder
    le = LabelEncoder()

    # Apply LabelEncoder to each column and transform back to DataFrame
    for col in cols:
        combined_df[col] = le.fit_transform(combined_df[col])

    # Split the combined data back into train and test sets
    df_train[cols] = combined_df.iloc[: len(df_train), :]
    df_test[cols] = combined_df.iloc[len(df_train) :, :]

    df_train["Purchase"] = np.log1p(df_train["Purchase"])

    df_train.to_csv(dataset_train.path + ".csv", index=False)
    df_test.to_csv(dataset_test.path + ".csv", index=False)


@component(
    packages_to_install=["pandas", "gcsfs", "scikit-learn", "numpy"],
    output_component_file="basic_preprocessing.yaml",
    base_image="python:3.9",
)
def basic_preprocessing(
    bucket_URI: str,
    folder: str,
    train: str,
    test: str,
    dataset_train: Output[Dataset],
    dataset_test: Output[Dataset],
):

    import pandas as pd

    df_train_uri = "".join([bucket_URI, folder, train])
    df_test_uri = "".join([bucket_URI, folder, test])

    df_train = pd.read_csv(df_train_uri)
    df_test = pd.read_csv(df_test_uri)

    df_train["Stay_In_Current_City_Years"] = df_train[
        "Stay_In_Current_City_Years"
    ].str.replace("+", "")
    df_train["Stay_In_Current_City_Years"] = df_train[
        "Stay_In_Current_City_Years"
    ].astype(int)

    df_test["Stay_In_Current_City_Years"] = df_test[
        "Stay_In_Current_City_Years"
    ].str.replace("+", "")
    df_test["Stay_In_Current_City_Years"] = df_test[
        "Stay_In_Current_City_Years"
    ].astype(int)

    ## Dropping User_id and Product_ID
    df_train = df_train.drop("User_ID", axis=1)
    df_test = df_test.drop("User_ID", axis=1)
    df_train = df_train.drop("Product_ID", axis=1)
    df_test = df_test.drop("Product_ID", axis=1)

    df_train = df_train.drop("Product_Category_3", axis=1)
    df_test = df_test.drop("Product_Category_3", axis=1)

    ## Imputing missing values with mode
    df_train["Product_Category_2"].mode()[0]
    df_train["Product_Category_2"] = df_train["Product_Category_2"].fillna(
        df_train["Product_Category_2"].mode()[0]
    )

    df_test["Product_Category_2"].mode()[0]
    df_test["Product_Category_2"] = df_test["Product_Category_2"].fillna(
        df_test["Product_Category_2"].mode()[0]
    )

    df_train.to_csv(dataset_train.path + ".csv", index=False)
    df_test.to_csv(dataset_test.path + ".csv", index=False)


@component(
    packages_to_install=["pandas", "gcsfs", "scikit-learn"],
    output_component_file="train_validation_test_split.yaml",
    base_image="python:3.9",
)
def train_validation_test_split(
    df_train: Input[Dataset],
    dataset_train: Output[Dataset],
    dataset_valid: Output[Dataset],
    validation_size: float = 0.2,
):

    import pandas as pd
    from sklearn.model_selection import train_test_split

    df_train = pd.read_csv(df_train.path + ".csv")

    df_train, df_valid = train_test_split(
        df_train, test_size=validation_size, random_state=42
    )

    df_train.to_csv(dataset_train.path + ".csv", index=False)
    df_valid.to_csv(dataset_valid.path + ".csv", index=False)
