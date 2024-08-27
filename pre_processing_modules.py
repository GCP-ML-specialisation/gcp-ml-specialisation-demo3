from kfp.v2.dsl import component, Input, Output, Dataset
import configparser
config = configparser.ConfigParser()
config.read("../config.ini")

@component(
    packages_to_install=["pandas", "gcsfs", "scikit-learn", "numpy", "scipy", "numpy", "imbalanced-learn", "imblearn"],
    output_component_file="output_files/feature_engineering.yaml",
    base_image="python:3.11",
)
def data_transformation(
    df: Input[Dataset],
    dataset: Output[Dataset],
    ):

    import pandas as pd
    from imblearn.over_sampling import SMOTENC
    from sklearn import preprocessing
    import numpy as np

    df = pd.read_csv(df.path + ".csv")

    df['Attrition'] = df['Attrition'].map(lambda x : 0 if x == 'no' else 1) 
    df['MaritalStatus'] = df['MaritalStatus'].map({'single': 0, 'married': 1, 'divorced': 2})
    num_bins=5
    
    binned_df = df.copy()
    columns = df.select_dtypes(include=['int64'])
    for col in columns:
        col_range = df[col].max() - df[col].min()
        if col_range > 5:
            # Calculate quantile-based bins
            _, bins = pd.qcut(df[col], q=num_bins, duplicates='drop', retbins=True)
            # Adjust bins to integer values
            bins = np.floor(bins).astype(int)
            bins[-1] = bins[-1] + 1  # Ensure the last bin is inclusive
            # Create labels based on bin edges
            labels = [f'{bins[i]}<=x<{bins[i+1]}' for i in range(len(bins)-1)]
            # Apply binning and replace original column
            binned_df[col] = pd.cut(df[col], bins=bins, labels=labels, right=False, include_lowest=True)
    df = binned_df
    print(df.head())
    df.drop(columns=['Gender','StandardHours', 'Over18', 'EmployeeCount', 'EmployeeNumber', 'BusinessTravel'], inplace=True)
    df['Department'].value_counts()
    department_dummies = pd.get_dummies(df['Department'], prefix='Department', dtype=float) # why now we used one-hot encoding and not label encoding?
    df.drop(columns=['Department'], inplace=True)
    df = pd.concat([df,department_dummies], axis=1)
    
    for field in ['JobRole', 'EducationField']:
        lb = preprocessing.LabelBinarizer()
        new_data = lb.fit_transform(df[field])
        binary_df = pd.DataFrame(new_data, columns=[f"{field}_{cls}" for cls in lb.classes_])
        df = pd.concat([df.drop(columns=[field]), binary_df], axis=1)


    # OverTime

    df['OverTime'] = df['OverTime'].map({'yes': 1, 'no': 0})
    print("Ciao we about to SMOTEEEE")
    desired_minority_count = int(0.3 * 1233)
    sampling_strategy = {0: 1233, 1: desired_minority_count}
    X_train = df.drop(columns=['Attrition'])
    y_train = df['Attrition']
    print(X_train.dtypes)
    smotenc = SMOTENC(categorical_features="auto", sampling_strategy=sampling_strategy, random_state=42)
    X_train_resampled, y_train_resampled = smotenc.fit_resample(X_train, y_train)

    # Combine resampled features with target
    df = pd.concat([pd.DataFrame(X_train_resampled, columns=X_train.columns), pd.DataFrame(y_train_resampled, columns=['Attrition'])], axis=1)
    # Education Field and JobRole

    
    

    df.to_csv(dataset.path + ".csv", index=False)


@component(
    packages_to_install=["pandas", "gcsfs", "scikit-learn", "numpy"],
    output_component_file="output_files/basic_preprocessing.yaml",
    base_image="python:3.11",
)
def basic_preprocessing(
    BUCKET_URI: str,
    FILE: str,
    dataset: Output[Dataset],
):

    import pandas as pd

    df_uri = "".join([BUCKET_URI, FILE])
    

    df = pd.read_csv(df_uri)
    categorical_columns = df.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        df[column] = df[column].str.replace(' ', '_').str.lower()
    
    df.Department = df.Department.map(lambda x : "research_and_development" if x == 'research_&_development' else x) 
    

    df.to_csv(dataset.path + ".csv", index=False)
