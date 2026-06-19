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

print(data.shape)

# Drop instances that violates logical intuition and business logic. 
loan_data = data.loc[~((data['age'] > 70) | (data['work_experience'] > 45))].copy()
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

# Now check if variables specified for binning exist in the dataset
for var in binning_config.keys():
    if var not in loan_data.columns:
        raise ValueError(f"Variable '{var}' specified for binning not found in the dataset.") 
    
