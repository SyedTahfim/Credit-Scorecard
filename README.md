# <img src="https://cdn-icons-png.flaticon.com/128/11315/11315900.png" width="20" /> Credit Scorecard Development
Credit scorecards quantify borrowers' willingness and ability to repay, making them critical tools in lending decisions and loan origination across banks and fintech platforms. This repository documents the complete credit scorecard development workflow, from data preparation, exploratory data analysis, binning strategy for WOE calculation, feature selection and transforming probability of defaults into credit scores.

- [Data Introduction](@DataIntroduction)
- [EDA Insight](@EDAInsight)

# Data Introduction
The data for this project was collected from Kaggle. The original dataset included 45,000 observations and 13 variables. Applied loan amount and interest rate were discarded before doing any analysis because they are terms of a loan, not risk indicators associated with borrower's willingness and ability to pay. Also, typically applied loan amount has a linear relationship with income, which may introduce multicollinearity in the model. 
Institutional lenders typically establish lending policies that restrict credit facilities to borrowers aged 65 and above, or those approaching or in retirement, due to heightened risk of income reduction or loss of stable income sources. Therefore, we have excluded borrowers aged above 70 years and work experience above 45 years. Exclusion of such borrowers resulted a dataset of 44,975 observations. 

The rest of variables are as follows:
- Age
- Gender
- Education Level
- Income
- Work Experience
- Home Ownership Status
- Loan Intent
- Loan to Income Ratio
- Credit History Length
- Previous Loan Defaults on File
- __Loan Status__-Target Variable (Accepted/Rejected) 

# EDA Insight
- __Class Imbalance__: Target variable is imbalanced with 77.8% rejected loans and 22.2% approved loans in the dataset, with no missing values.  
- __Demographic Patterns__: Income peaks around age 40 then stabilizes/declines; approved loans concentrate among younger borrowers with lower incomes, while older adults show higher credit scores with no association with income.
- __Correlated Features__: Age, work experience, and credit history length are strongly intercorrelated that is likely to result multicollinearity; older borrowers systematically show longer credit histories and higher credit scores.
- __Categorical Significance__: Chi-square tests reveal three statistically significant categorical variables; encompassing previous loan defaults, home ownership status, and loan intent; education and gender show no association with loan approval. 
- __Variable Selection Implication__: Education and gender flagged for further WoE/IV analysis before potential exclusion; previous loan defaults emerges as the strongest univariate categorical predictor, suggesting high predictive value for PD modeling.

 <p align="center">
    <img src="https://github.com/SyedTahfim/Credit-Scorecard/blob/main/pair_plot.png" width="60%" alt="Image 1 Description" style="float: left; margin-right: 2%;">
 </p>

## Rest is Work in Progress


