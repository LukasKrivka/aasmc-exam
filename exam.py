import pandas as pd
import statsmodels.formula.api as smf

if __name__ == '__main__':

    df = pd.read_csv('data/adult.csv')

    # #View dataset and exoplore different attributes and
    # print(df.head(20))
    #
    # for c in df.columns:
    #     print(c)
    #     print(df[c].describe())
    #     print('unique: {}'.format(df[c].unique()))
    #     print()

    # preprocessing to fit into regression
    df.rename(columns={'hours.per.week': 'hours_per_week'}, inplace=True)
    df.replace(to_replace='Asian-Pac-Islander', value='Asian_Pac_Islander', inplace=True)
    df.replace(to_replace='Amer-Indian-Eskimo', value='Amer_Indian_Eskimo', inplace=True)
    df.drop(df[df['native.country'] == '?'].index, inplace=True)
    df.drop(df[df['occupation'] == '?'].index, inplace=True)
    df.drop(df[df['workclass'] == '?'].index, inplace=True)
    df['income_num'] = pd.Categorical(df['income']).codes
    # pd.concat([df, pd.get_dummies(df['race'], pd.get_dummies(df[''])])
    print(df.head(30))

    # obtaining regression results
    formula_str = 'income_num ~ age + sex + hours_per_week + workclass + education'# + occupation + race'
    # formula_str = 'income_num ~ age + sex + hours_per_week + workclass + race + occupation + race'
    # formula_str = 'income_num ~ sex + hours_per_week + race + workclass + occupation + education'
    # formula_str = 'income_num ~ education + age + sex'
    results = smf.logit(formula=formula_str, data=df).fit()
    print(results.summary())

    # contingency table and chi-squared testing:
    # crosstable = pd.crosstab(index=df['race'], columns=df['sex'])
    # print(crosstable)


