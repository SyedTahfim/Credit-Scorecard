"""
This script performs data processing task on raw train and test data:
1. Keep individual applications issued from 2010 onward.
2. Derive features and the default flag.
3. Drop conflicting rows and irrelevant columns.
"""

import pandas as pd
import numpy as np 
import logging 

log = log = logging.getLogger(__name__)


def preprocess_data(data: pd.DataFrame, prep_cfg: dict) -> pd.DataFrame:
    """
    Preprocess the train and test datasets based on the provided configuration.

    Parameters:
    - prep_cfg: Dictionary containing the preprocessing configuration.
 
    Returns:
    - DataFrame containing the preprocessed data.
    """ 
    # Check if the data is loaded successfully
    if data.empty:
        raise ValueError(f"The data is empty or could not be loaded.")
    
    # The scorecard is for individual loan applications. Dropping joint loan applications.  
    data = data.loc[data['application_type'] != "JOINT"]

    # Drop Application type and joint loan relation columns 
    data = data.drop(columns=prep_cfg['appl_cols'])

    # Drop Loans Issued Before 2010 
    for col in prep_cfg['date_cols']:
        data[col] = pd.to_datetime(data[col], format="%b-%Y", errors="coerce")
    data = data.loc[data['issue_d'].dt.year >= 2010]

    """
    We are dropping columns including missing values more than 90 percent because the missingness is probably casued by some system issue. 
    """
    sparse_cols = data.columns[data.isna().mean() > 0.90]
    data = data.drop(columns=sparse_cols)

    # Fill missing values of columns where missingness is casued by absence of delinquency or derogoratory records 
    cols_to_fill = prep_cfg["cols_to_fill"]
    data[cols_to_fill] = data[cols_to_fill].fillna(0)

    # Derive/Modify Variables
    """
    we can derive some new variables to increase model's discrimatory power and avoid loosing valuable information due to multicollinearity. 
    """
    # Trasnform the employment length column to numeric values
    data['emp_length'] = data['emp_length'].str.strip().str.lower().str.replace(r'[^\d]', '', regex=True) 

    # Clean Term Variable
    data['term'] = data['term'].str.strip().str.lower().str.replace(r'[^\d]', '', regex=True)
    data['term'] = data['term'].replace({'36':'three_year','60':'five_year'})

    # Loan to income ratio > loan amount / annual income
    data["loan_to_annl_inc"] = (data["loan_amnt"] / data["annual_inc"].replace(0, np.nan)).round(2)
    # The latest payment date is 2016-01-01, so we are setting the observation date to 2016-01-01
    data['obs_d'] = pd.to_datetime(prep_cfg['obs_date'])
    
    # The difference between the earliest credit line and issue date is the length of credit history
    data['credit_history_length'] = (data['issue_d'].dt.year - data['earliest_cr_line'].dt.year) * 12 + \
                                    data['issue_d'].dt.month - data['earliest_cr_line'].dt.month
    data['mnths_loan_sanct'] = (data['obs_d'].dt.year - data['issue_d'].dt.year)*12 + \
                                            (data['obs_d'].dt.month - data['issue_d'].dt.month)

    # Dropping loans with less than a year maturity 
    data = data.loc[data['mnths_loan_sanct'] > 12].copy() 

    # Months Overdue 
    data['MOD'] = (data['obs_d'] - data['next_pymnt_d']).dt.days / 30.25 

    # Credit Default Flag
    data['default_flag'] = 0
    data.loc[(data['loan_status'] == "Charged Off") | (data['loan_status'] == "Default") | 
    (data['loan_status'] == "Does not meet the credit policy. Status:Charged Off"),'default_flag'] = 1
    data.loc[(data['loan_status'] == "Late (31-120 days)") & 
        (data['MOD'] >=4),"default_flag"] = 1
    

    # Drop charged-off loans with zero overdue months
    data = data.loc[~((data["loan_status"] == "Charged Off") & (data["MOD"] == 0))]

   # Drop rows with conflicting business logic
    conflict_delinq = (
        (data["mths_since_last_delinq"] != 0)
        & (data["mths_since_last_delinq"] < 24)
        & (data["delinq_2yrs"] == 0)
    )
    conflict_revol = (data["revol_util"] == 0) & (data["revol_bal"] != 0)
    data = data.loc[~(conflict_delinq | conflict_revol)]

    # Dropping rare categories in home ownership column
    data = data[~data['home_ownership'].isin(['OTHER', 'NONE', 'ANY'])] 

    final_df = data.drop(columns = prep_cfg['drop_columns'])

    return final_df



