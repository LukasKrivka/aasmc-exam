import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
# import statsmodels.discrete.discrete_model as sm
import statsmodels.api as sm
import matplotlib.pyplot as plt

from scipy.stats import chi2, norm, kstest, kruskal, spearmanr
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from itertools import combinations


def loglike(self, params):
    a, b = params
    y_pred = np.dot(X, params)
    sigma = np.std(y - y_pred)
    return np.sum(norm.logpdf(y, loc=y_pred, scale=sigma))


def chi_test_independence(data: pd.DataFrame, att1: str, att2: str):
    """function to get test statistic and p-value to test significance of association
     between two categorical variables"""

    crosstable = pd.crosstab(index=data[att1], columns=data[att2])
    expected_count = crosstable.copy()
    crosstable['row_total'] = [crosstable.loc[x].sum() for x in crosstable.index]
    crosstable.loc['column_total'] = [crosstable[x].sum() for x in crosstable.columns]
    overall_total = crosstable.loc['column_total', 'row_total']
    for i in expected_count.index:
        for c in expected_count.columns:
            expected_count.loc[i, c] = crosstable.loc[i, 'row_total'] * crosstable.loc['column_total', c] / overall_total

    chi_statistic_df = (crosstable.drop(index=['column_total'], columns=['row_total']) - expected_count)**2 / expected_count
    chi_statistic = chi_statistic_df.to_numpy().sum()
    deg_freedom = (len(crosstable.index) - 2) * (len(crosstable.columns) - 2)  # subtract two because crosstable contains totals
    p_value = chi2.cdf(chi_statistic, deg_freedom)

    return chi_statistic, p_value


def kruskal_test(data: pd.DataFrame, att1: str, att2: str):
    num = data[att1]
    cat = data[att2]
    stat, p_value = kruskal(*[num[cat == value] for value in cat.unique()])

    return stat, p_value


sex = {1: "Male", 2: "Female"}

race = {1: "White", 2:"African-American", 3:"American-Indian", 4:"Alaska Native", 5:"Am. Indian and Alaska Nat.",
        6: "Asian", 7:"Native Hawaiian or Other Pacific Islander", 8: "Other Races (alone)",
        9: "Two or More Races"}

cow = {1: "Private (for profit)", 2:"Private (non-profit)", 3: "Local gov.", 4: "State gov.", 5:"Federal gov.",
       6: "Self-employed (not inc.)", 7: "Self-employed (inc.)", 8:"Without Pay", 9: "Unemployed" }

cit = {1: "Born in US", 2: "Born in unincorporated territory", 3:"Born abroad (US parents)",
       4: "Citizen by Naturalisation", 5:"Not a citizen"}

mar = {1: "Married", 2:"Widowed", 3:"Divorced", 4:"Separated", 5:"Never Maried"}

schl = {1: "No schooling completed", 2: "Nursery school, preschool", 3: "Kindergarten", 4: "Grade 1", 5: "Grade 2",
        6: "Grade 3", 7: "Grade 4", 8: "Grade 5", 9: "Grade 6", 10: "Grade 7", 11: "Grade 8", 12: "Grade 9",
        13: "Grade 10", 14: "Grade 11", 15: "12th grade - no diploma", 16: "Regular high school diploma",
        17: "GED or alternative credential", 18: "Some college, but less than 1 year",
        19: "1 or more years of college credit, no degree", 20: "Associate's degree", 21: "Bachelor's degree",
        22: "Master's degree", 23: "Professional degree beyond a bachelor's degree", 24: "Doctorate degree"}

waob =  {1: "US state", 2: "PR and US Island Areas", 3: "Latin America", 4: "Asia", 5: "Europe", 6: "Africa",
         7: "Northern America", 8: "Oceania and at Sea"}

relshipp = {20: "Reference person", 21: "Opposite-sex husband/wife/spouse", 22: "Opposite-sex unmarried partner",
            23: "Same-sex husband/wife/spouse", 24: "Same-sex unmarried partner", 25: "Biological son or daughter",
            26: "Adopted son or daughter", 27: "Stepson or stepdaughter", 28: "Brother or sister",
            29: "Father or mother", 30: "Grandchild", 31: "Parent-in-law", 32: "Son-in-law or daughter-in-law",
            33: "Other relative", 34: "Roommate or housemate", 35: "Foster child", 36: "Other nonrelative",
            37: "Institutionalized group quarters population", 38: "Noninstitutionalized group quarters population"}

occp = {'MGR': "Management Occupations", 'BUS': "Business Occupations", 'FIN': "Financial Operations Occupations",
        'CMM': "Computer and Mathematical Occupations", 'ENG': "Architecture and Engineering Occupations",
        'SCI': "Life, Physical, and Social Science Occupations", 'CMS': "Community and Social Service Occupations",
        'LGL': "Legal Occupations", 'EDU': "Educational Instruction and Library Occupations",
        'ENT': "Arts, Design, Entertainment, Sports, and Media Occupations",
        'MED': "Healthcare Practitioners and Technical Occupations",
        'HLS': "Healthcare Support Occupations", 'PRT': "Protective Service Occupations",
        'EAT': "Food Preparation and Serving Related Occupations",
        'CLN': "Building and Grounds Cleaning and Maintenance Occupations",
        'PRS': "Personal Care and Service Occupations", 'SAL': "Sales and Related Occupations",
        'OFF': "Office and Administrative Support Occupations", 'FFF': "Farming, Fishing, and Forestry Occupations",
        'CON': "Construction Occupations", 'EXT': "Extraction Occupations",
        'RPR': "Installation, Maintenance, and Repair Occupations", 'PRD': "Production Occupations",
        'TRN': "Transportation and Material Moving Occupations", 'MIL': "Military Specific Occupations",
        'UEP': "Unemployed, With No Work Experience In The Last 5 Years Or Earlier Or Never Worked"}

encoding = {'SEX': sex, 'RAC1P': race, 'SCHL': schl, 'COW': cow, 'MAR': mar, 'RELSHIPP': relshipp, 'WAOB': waob, 'CIT': cit, 'OCCP': occp}

if __name__ == '__main__':

    df = pd.read_csv('data/folktables_data/CA_new.csv')
    for k, v in encoding.items():
        df[k] = df[k].map(v)
    df.rename(columns={'RAC1P': 'RACE', 'AGEP': 'AGE', 'SCHL': 'EDUCATION', 'MAR': 'FAMILY_STATUS',
                       'CIT': 'CITIZENSHIP', 'COW': 'WORKCLASS', 'RELSHIPP': 'RELATIONSHIP',
                       'WAOB': 'BIRTH_PLACE', 'WKHP': 'WORK_HOURS', 'INTP': 'CAPITAL_TAX',
                       'OCCP': 'OCCUPATION', 'PINCP': 'INCOME'}, inplace=True)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    # print(df.head(30))

    # testing independence of selected features
    # categorical = ['SEX', 'RAC1P', 'SCHL', 'COW', 'OCCP', 'MAR', 'POBP', 'RELSHIPP', 'ESR', 'WAOB', 'MSP', 'CIT']
    # numeric = ['AGEP', 'WKHP', 'INTP']
    categorical = ['SEX', 'RACE', 'EDUCATION', 'WORKCLASS', 'FAMILY_STATUS', 'OCCUPATION',
                   'RELATIONSHIP', 'BIRTH_PLACE', 'CITIZENSHIP']
    numeric = ['AGE', 'WORK_HOURS', 'CAPITAL_TAX']

    # testing normality of numeric features to determine whether to use parametric or non-parametric test for independence
    for i in numeric:
        stat, p_value = kstest(df[i], 'norm')
        print('p-value for normality of {}: {}'.format(i, p_value))

    # create matrix for p-values for dependence tests, independence_df captures all features, for the report, individual df are made for different data types
    features = categorical+numeric
    independence_df = pd.DataFrame(index=features, columns=features)
    for i in features:
        independence_df.loc[i, i] = "-"

    cat_p = pd.DataFrame(index=categorical, columns=categorical)
    num_p = pd.DataFrame(index=numeric, columns=numeric)
    mix_p = pd.DataFrame(index=categorical, columns=numeric)
    corr_df = pd.DataFrame(index=numeric, columns=numeric)

    for i, j in combinations(features, 2):
        # testing dependence between two categorical features
        if i in categorical and j in categorical:
            chi, p = chi_test_independence(data=df, att1=i, att2=j)
            independence_df.loc[i, j] = p
            independence_df.loc[j, i] = p
            cat_p.loc[i, j] = p
            cat_p.loc[j, i] = p
        # testing dependence between categorical and numeric features
        if i in categorical and j in numeric:
            t, p = kruskal_test(data=df, att1=j, att2=i)
            independence_df.loc[i, j] = p
            independence_df.loc[j, i] = p
            mix_p.loc[i, j] = p
        if i in numeric and j in categorical:
            t, p = kruskal_test(data=df, att1=i, att2=j)
            independence_df.loc[i, j] = p
            independence_df.loc[j, i] = p
            mix_p.loc[j, i] = p
        # testing dependence between two numeric features
        if i in numeric and j in numeric:
            r = spearmanr(df[i], df[j])
            independence_df.loc[i, j] = r[1]
            independence_df.loc[j, i] = r[1]
            num_p.loc[j, i] = r[1]
            num_p.loc[i, j] = r[1]
            corr_df.loc[i, j] = r[0]
            corr_df.loc[j, i] = r[0]

    cat_p.fillna('-', inplace=True)
    num_p.fillna('-', inplace=True)
    corr_df.fillna(1, inplace=True)

    # print(independence_df.to_latex())
    print(cat_p.to_latex())
    print(num_p.to_latex())
    print(mix_p.to_latex())
    print(corr_df.to_latex())

    # TODO: REGRESSION (todo is made for highlighting)
    """obtaining regression results
    Using k-1 dummy variables and centered and standerdized numeric features"""
    X = (df[numeric] - df[numeric].mean()) / df[numeric].std()
    X = pd.concat([X, pd.get_dummies(df[categorical].drop(columns=['WORKCLASS']))], axis=1) # , 'RELATIONSHIP', 'BIRTH_PLACE'
    # the R^2 including all is 0.465, excluding OCCP 0.398, excluding POBP 0.462, excluding both 0.386,, + excluding WORKCLASS 0.378 (with HC1)
    print(df['INCOME'].describe())
    y = np.log(df['INCOME'])
    y = (y - y.mean()) / y.std()
    X = add_constant(X)
    # Dropping categorical dummies manually, for better interpretation of results (compared to the base case that is droped)
    X.drop(columns=['SEX_Male', 'RACE_White', 'BIRTH_PLACE_US state', 'RELATIONSHIP_Biological son or daughter',# 'WORKCLASS_Without Pay',
                    'FAMILY_STATUS_Never Maried', 'EDUCATION_No schooling completed', 'CITIZENSHIP_Born in US'], inplace=True)
    model = sm.OLS(y, X)
    results = model.fit(cov_type='HC2')
    print(results.summary())

    # plotting income to make the point that log transformation makes it more like normally distributed
    # plt.hist(df['INCOME'])
    # plt.show()
    # plt.hist(y)
    # plt.show()
    #
    # plotting the residuals against y and predictors for the report (testing homoskedasticity and endogenity)
    # plt.scatter(y, results.resid, alpha=0.3)
    # plt.xlabel('Y')
    # plt.ylabel('Residuals')
    # plt.savefig('Residuals_INCOME')
    # plt.show()
    # plt.scatter(X['AGE'], results.resid, alpha=0.3)
    # plt.xlabel('AGE')
    # plt.ylabel('Residuals')
    # plt.savefig('Residuals_AGE')
    # plt.show()
    # plt.scatter(X['WORK_HOURS'], results.resid, alpha=0.3)
    # plt.xlabel('WORK_HOURS')
    # plt.ylabel('Residuals')
    # plt.savefig('Residuals_WORK_HOURS')
    # plt.show()
    # plt.scatter(X['CAPITAL_TAX'], results.resid, alpha=0.3)
    # plt.xlabel('CAPITAL_TAX')
    # plt.ylabel('Residuals')
    # plt.savefig('Residuals_CAPITAL_TAX')
    # plt.show()
    #
    # # testing normality of residuals:
    # stat, p_value = kstest(results.resid, 'norm')
    # print('p-value for normality of {}: {}'.format('residuals', p_value))
    #
    # # testing VIF:
    # vif_data = pd.DataFrame()
    # vif_data["Regressors"] = X.columns
    # vif_data["VIF"] = [variance_inflation_factor(X.values, i)
    #                    for i in range(len(X.columns))]
    # print(vif_data)
    #
    # # testing homoskedasticity:
    # bp_test = het_breuschpagan(results.resid, X)
    # print('Breusch-Pangan test:')
    # print('LM-test p-value:', bp_test[1])
    # print('F-test p-value:', bp_test[3])
    #
    # # getting insignificant attributes for report
    result_summary = results.summary().tables[1].data
    result_summary = pd.DataFrame(result_summary, columns=['Predictor', 'coef', 'std err', 't', 'p-value', 'Conf.',  'Int.'])
    result_summary.set_index(result_summary.iloc[:, 0], inplace=True, drop=True)
    result_summary = result_summary.iloc[1:, [1, 2, 4]]
    result_summary['p-value'] = pd.to_numeric(result_summary['p-value'])
    print('insignificant coefficients')
    print(result_summary[result_summary['p-value'] > 0.05].to_latex())
    result_summary = result_summary[result_summary['p-value'] < 0.05]
    result_summary['coef'] = pd.to_numeric(result_summary['coef'])
    result_summary['coef_abs'] = result_summary['coef'].abs()
    print('top 10 coefficients')
    print(result_summary.sort_values('coef_abs', ascending=False).head(10).drop(columns=['coef_abs']).to_latex())
