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

--------------------------------------------------------------------------------
________________________________________________________________________________
Another approach is to predict whether a person makes above 50K using logistic regression
With all the features, there is an issue of (most likely) multicollinearity.
We can use LASSO regression to address this and add chi-squared test or t-test for dependence between
variables that are identified as insignificant by the constrained regression