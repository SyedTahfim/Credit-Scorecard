# Import Libraries
import os
import pandas as pd
import numpy as np 
import warnings 
warnings.filterwarnings('ignore') 


file_path = "/mnt/Docs/Git_Repo/Credit_Scorecard/Data/raw/loan_data_renamed.csv"

if os.path.exists(file_path):
    try:
        data = pd.read_csv(file_path)
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error: An error occurred while loading the data: {e}")
    else:
        print(f"Error: The file at {file_path} was not found. Please check the path and try again.")

print(data.drop(columns="loan_status").select_dtypes(include=[np.number]).columns) 


def bin_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bins a list of categorical variables in a DataFrame according to the specified binning configuration.
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    Returns:
    pd.DataFrame: A DataFrame with the binned variables.
    """

    # Drop instances that violates logical intuition and business logic. 
    loan_data = df.loc[~((df['age'] > 70) | (df['work_experience'] > 45))].copy()
    loan_data = loan_data.drop(columns=['loan_amnt','loan_int_rate'])
    print(loan_data.shape)

    binning_config = {
    "age": {
        "bins": [20,35,45,65],
        "labels": ["young_adults","prime_adults","middle_aged"]
    },
    "income": {
        "bins": [0,40,60,120,180,np.inf],
        "labels": ['low','lower-middle','middle','upper-middle','high']
    },
    "work_experience": {
        "bins": [0, 3, 7, 12, 20, np.inf],
        "labels": ["entry_level", "early_career", "mid_level", "established","veteran"]
    },
    "loan_to_income_ratio": {
        "bins": [0,0.25,0.45,1],
        "labels": ["low","medium","high"]
    },
    "credit_history_length": {
        "bins": [2,5,10,np.inf],
        "labels": ["2-5","5-10","10+"]
    },
    }

    # Selecting only numerical columns for binning.
    num_cols = list(loan_data.drop(columns="loan_status").select_dtypes(include=[int,float]).columns)

    # Apply Binning based on the defined bins and labels. 
    for var in num_cols:
        if var in binning_config.keys():
            bins = binning_config[var]['bins']
            labels = binning_config[var]['labels']
            loan_data[var + "_binned"] = pd.cut(loan_data[var], bins=bins, labels=labels, 
                                                include_lowest=True)
        else:
            raise ValueError(f"Bins and Labels not found for variable '{var}'.")
    return loan_data.drop(columns=num_cols)


loan_data_binned = bin_variable(data)

print(loan_data_binned.columns)


