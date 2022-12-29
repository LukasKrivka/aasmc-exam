import pandas as pd
import statsmodels.formula.api as smf

from scipy.stats import chi2
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
    deg_freedom = (len(crosstable.index) - 2) * (len(crosstable.columns) - 2)  # we subtract two because crosstable contains totals
    p_value = chi2.cdf(chi_statistic, deg_freedom)

    return chi_statistic, p_value


if __name__ == '__main__':

    df = pd.read_csv('data/adult.csv')

    """View dataset and exoplore different attributes and"""
    # print(df.head(20))
    #
    # for c in df.columns:
    #     print(c)
    #     print(df[c].describe())
    #     print('unique: {}'.format(df[c].unique()))
    #     print()

    """preprocessing to fit into regression"""
    df.rename(columns={'hours.per.week': 'hours_per_week'}, inplace=True)
    df.replace(to_replace='Asian-Pac-Islander', value='Asian_Pac_Islander', inplace=True)
    df.replace(to_replace='Amer-Indian-Eskimo', value='Amer_Indian_Eskimo', inplace=True)
    df.drop(df[df['native.country'] == '?'].index, inplace=True)
    df.drop(df[df['occupation'] == '?'].index, inplace=True)
    df.drop(df[df['workclass'] == '?'].index, inplace=True)
    df['income_num'] = pd.Categorical(df['income']).codes
    # print(df.head(30))

    """obtaining regression results
    Other formula strings are there only for debugging multicollinearity"""
    # formula_str = 'income_num ~ age + sex + hours_per_week + workclass + race + occupation + race'
    # formula_str = 'income_num ~ sex + hours_per_week + race + workclass + occupation + education'
    # formula_str = 'income_num ~ education + age + sex'
    formula_str = 'income_num ~ age + sex + hours_per_week + workclass + education'# + occupation + race'
    results = smf.logit(formula=formula_str, data=df).fit()
    print(results.summary())

    # chi-squared testing for all categorical variables combinations:
    categorical = ['sex', 'race', 'education', 'workclass', 'occupation']
    independence_df = pd.DataFrame(index=categorical, columns=categorical)
    for i in categorical:
        independence_df.loc[i, i] = "-"
    for i, j in combinations(categorical, 2):
        chi, p = chi_test_independence(data=df, att1=i, att2=j)
        independence_df.loc[i, j] = p
        independence_df.loc[j, i] = p

    print(independence_df)
