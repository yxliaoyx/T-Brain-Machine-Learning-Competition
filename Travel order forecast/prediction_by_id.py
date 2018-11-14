import numpy as np
import pandas as pd

df_test = pd.read_csv('testing-set.csv')
df_train = pd.read_csv('training-set.csv')

df_train_deal = df_train[df_train['deal_or_not'] == 1]
df_train_not_deal = df_train[df_train['deal_or_not'] == 0]

not_deal = df_test['order_id'] > df_train_deal['order_id'].max()

df_test_copy = df_test.copy()
df_test_copy['deal_or_not'] = np.random.rand(len(df_test))
df_test_copy['deal_or_not'] = (df_test_copy['deal_or_not'] < (0.5 / (1 - len(df_test[not_deal]) / len(df_test)))) * 1

df_test_copy[not_deal] = 0
df_test['deal_or_not'] = df_test_copy['deal_or_not']

df_test.to_csv('submission_prediction_by_id.csv', index=False)
print(df_test)
print(df_test['deal_or_not'].mean())
