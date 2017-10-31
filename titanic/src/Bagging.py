# coding=UTF-8
import pandas as pd

result1 = pd.read_csv('../input/submit2.csv', index_col=0)
result2 = pd.read_csv('../input/bagging_submit2.csv', index_col=0)
result3 = pd.read_csv('../input/random_forest_gridsearch_submit.csv', index_col=0)
result4 = pd.read_csv('../input/bagging_submit.csv', index_col=0)
result5 = result1 + result2 + result3 + result4
result5['Survived'] = result5['Survived'].apply(lambda x: 1 if x>=2 else 0)
# print result5.head(10)
# print type(result5)
result5.to_csv('../input/bagging_submit3.csv', index=True)


