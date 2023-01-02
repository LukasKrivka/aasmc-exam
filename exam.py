import pandas as pd
import statsmodels.formula.api as smf
# import statsmodels.discrete.discrete_model as sm
import statsmodels.api as sm

from statsmodels.tools.tools import add_constant
from scipy.stats import chi2, kstest, kruskal, spearmanr
from itertools import combinations


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


if __name__ == '__main__':

    """OLD DATASET"""
    # df = pd.read_csv('data/adult.csv')
    #
    # """preprocessing to fit into regression"""
    # df.rename(columns={'hours.per.week': 'hours_per_week', 'capital.gain': 'capital_gain', 'capital.loss': 'capital_loss'}, inplace=True)
    # df.replace(to_replace='Asian-Pac-Islander', value='Asian_Pac_Islander', inplace=True)
    # df.replace(to_replace='Amer-Indian-Eskimo', value='Amer_Indian_Eskimo', inplace=True)
    # df.drop(df[df['native.country'] == '?'].index, inplace=True)
    # df.drop(df[df['occupation'] == '?'].index, inplace=True)
    # df.drop(df[df['workclass'] == '?'].index, inplace=True)
    # df['income_num'] = pd.Categorical(df['income']).codes
    # # print(df.head(30))
    #
    # # testing independence of selected features
    # categorical = ['sex', 'race', 'education', 'workclass', 'occupation']
    # numeric = ['age', 'hours_per_week', 'capital_gain', 'capital_loss']
    #
    # X = df[['age', 'hours_per_week', 'capital_gain', 'capital_loss']]
    # X = pd.concat([X, pd.get_dummies(df[['sex', 'education', 'occupation', 'race']])], axis=1)
    # y = df['income_num']
    # X = add_constant(X)
    # model = sm.Logit(y, X)
    # results = model.fit()
    # print(results.summary())

    df = pd.read_csv('data/folktables_data/CA_w_few_cols.csv')

    # testing independence of selected features
    categorical = ['SEX', 'RAC1P', 'SCHL', 'COW', 'OCCP', 'MAR', 'POBP', 'RELSHIPP', 'ESR', 'WAOB', 'MSP', 'CIT']
    numeric = ['AGEP', 'WKHP', 'INTP']

    # testing normality of numeric features to determine whether to use parametric or non-parametric test for independence
    for i in numeric:
        stat, p_value = kstest(df[i], 'norm')
        print('p-value for normality of {}: {}'.format(i, p_value))

    features = categorical+numeric
    independence_df = pd.DataFrame(index=features, columns=features)
    for i in features:
        independence_df.loc[i, i] = "-"

    for i, j in combinations(features, 2):
        # testing dependence between two categorical features
        if i in categorical and j in categorical:
            chi, p = chi_test_independence(data=df, att1=i, att2=j)
            independence_df.loc[i, j] = p
            independence_df.loc[j, i] = p
        # testing dependence between categorical and numeric features
        if i in categorical and j in numeric:
            t, p = kruskal_test(data=df, att1=j, att2=i)
            independence_df.loc[i, j] = p
            independence_df.loc[j, i] = p
        if i in numeric and j in categorical:
            t, p = kruskal_test(data=df, att1=i, att2=j)
            independence_df.loc[i, j] = p
            independence_df.loc[j, i] = p
        # testing dependence between two numeric features
        if i in numeric and j in numeric:
            r = spearmanr(df[i], df[j])
            independence_df.loc[i, j] = r[1]
            independence_df.loc[j, i] = r[1]

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    # print(independence_df)

    """obtaining regression results"""
    X = df[numeric]
    X = pd.concat([X, pd.get_dummies(df[categorical].drop(columns=['OCCP', 'POBP']).astype(str))], axis=1)
    # the R^2 including all is 0.465, excluding OCCP 0.398, excluding POBP 0.462, excluding both 0.391
    y = df['PINCP'] / 1000
    X = add_constant(X)
    model = sm.OLS(y, X)
    results = model.fit()
    print(results.summary())
