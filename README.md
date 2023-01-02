# aasmc-exam

Dataset: https://www.kaggle.com/datasets/uciml/adult-census-income

Question / Hypothesis:
Does this data indicate discrimination or racism?

Supporting hypothesis to be tested:
- is the attribute "race" significant as a predictor when explaining "income"
- is the attribute "gender" significant as a predictor when explaining "income"

Methods:
- Logistic Regression - to get the estimators of the predictors' coefficient
- t-Test - to test the significance of the coefficient (H0: coefficient = 0)

_________________________________________________________________________________
---------------------------------------------------------------------------------
Change the frame of the project / paper to dimensionality reduction
- we are interested in explaining what attributes influence whether a person makes 50k
- with categorical variables encoded as dummy variables, the number of features grows very fast
  - logistic regression including around half of attributes has issues with multicollinearity (singular matrix and no convergence)
- we can test their association (categorical x categorical) with chi-squared test of independence and remove highly correlated
- we can fit a regression and then from the estimators of the coefficients mean and std test their significance (do they differ from 0)
- we can fit the LASSO regression instead using regularization to get rid of less important variables
- we can also use PCA (or MCA) to find which attributes contribute to the most significant components

Second dataset(s): Folktables

Filtered by: 
AGEP > 16,  PINCP > 100, WKHP > 0, PWGTP (personal weight) >= 1

feature_names: 
'AGEP' # Age (Continuous)
'SEX' # Gender (Binary, 1: male, 2: female)
'WKHP' # Hours-per-week (Continuous)
'COW' # Workclass (Categorical, nominal)
'SCHL' # Educational attainment (Categorical, ordinal)
'MAR' # Marital status (Categorical, nominal)
'RAC1P' # Recoded detailed race code (Categorical, nominal)
'POBP' # Place of birth (native-country) (Categorical, nominal)
'OCCP' # Occupation (Categorical, nominal)
'INTP' # Interests, dividends and net rental income (Continuous) (capital gain - capital loss)
'RELSHIPP' # Relationship (Categorical, nominal)
'ESR' # Employment status recode (Categorical, nominal) (extra)
'WAOB' # World area of birth (Categorical, nominal) (extra)
'MSP' # Other marital status (Categorical, nominal) (extra)
'CIT' # Citizenship status (Categorical, nominal) (extra)

target_name:
'PINCP' # Total person's income (Continuous)
