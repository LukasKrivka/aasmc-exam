# aasmc-exam

Dataset: https://www.kaggle.com/datasets/uciml/adult-census-income

Question / Hypothesis:
Does this data indicate discrimination or racism?

Supporting hypothesis to be tested:
- is the attribute "race" significant as a predictor when explaining "income"
- is the attribute "gender" significant as a predictor when explaining "income"

Methods:
- Linear Regression - to get the estimators of the predictors' coefficient
- t-Test - to test the significance of the coefficient (H0: coefficient = 0)
- (PCA - to find out how much does "race" and "gender" contribute to the most significant dimensions)