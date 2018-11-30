# import numpy as np
# import pandas as pd
# import lightgbm as lgb
#
# df_test = pd.read_csv('testing-set.csv')
# df_train = pd.read_csv('training-set.csv')
#
# df_train_deal = df_train[df_train['deal_or_not'] == 1]
# df_train_not_deal = df_train[df_train['deal_or_not'] == 0]
#
# not_deal = df_test['order_id'] > df_train_deal['order_id'].max()
#
# df_test_copy = df_test.copy()
# df_test_copy['deal_or_not'] = np.random.rand(len(df_test))
# df_test_copy['deal_or_not'] = (df_test_copy['deal_or_not'] < (0.5 / (1 - len(df_test[not_deal]) / len(df_test)))) * 1
#
# df_test_copy[not_deal] = 0
# df_test['deal_or_not'] = df_test_copy['deal_or_not']
#
# df_test.to_csv('prediction_by_id.csv', index=False)
# # print(df_test)
# print(df_test['deal_or_not'].mean())
#
#
# train_prediction = pd.read_csv('train_prediction.csv')
#
# train_prediction_copy = train_prediction.copy()
# # print(train_prediction_copy[df_train['order_id'] > df_train_deal['order_id'].max()])
# train_prediction_copy[df_train['order_id'] > df_train_deal['order_id'].max()] = 0
# train_prediction['deal_or_not'] = train_prediction_copy['deal_or_not']
#
# # print(train_prediction)
# print(train_prediction['deal_or_not'].mean())
#
#
# df_submission = pd.read_csv('submission.csv')
#
# df_submission_copy = df_submission.copy()
# # print(df_submission_copy[not_deal])
# df_submission_copy[not_deal] = 0
# df_submission['deal_or_not'] = df_submission_copy['deal_or_not']
#
# # df_submission.to_csv('submission_prediction_by_id.csv', index=False)
# # print(df_submission)
# print(df_submission['deal_or_not'].mean())

import pandas as pd

df_test = pd.read_csv('cmp4_sample_submission.csv', dtype={'order_id': str})
# df_train = pd.read_csv('testing-set.csv', dtype={'order_id': str})
# print(df_train)
df_order = pd.read_csv('dataset/order.csv', dtype={'order_id': str, 'group_id': str})

leakage = pd.merge(df_order[['order_id', 'group_id']], df_test)

leakage['index'] = leakage.index

leakage_max_index = leakage[['group_id', 'index']].groupby('group_id').max().reset_index()
leakage_max_index = leakage_max_index.rename(columns={'index': 'max_index'})
leakage = pd.merge(leakage, leakage_max_index)

leakage_min_index = leakage[['group_id', 'index']].groupby('group_id').min().reset_index()
leakage_min_index = leakage_min_index.rename(columns={'index': 'min_index'})
leakage = pd.merge(leakage, leakage_min_index)

leakage['deal_or_not_leakage'] = (leakage['index'] - leakage['min_index']) / (
        leakage['max_index'] - leakage['min_index'])
leakage_copy = leakage.copy()
leakage_copy['deal_or_not_leakage'] = 0
leakage_copy[leakage['deal_or_not_leakage'] < 0.191] = 1
leakage['deal_or_not_leakage'] = leakage_copy['deal_or_not_leakage']

# print(leakage[['deal_or_not', 'deal_or_not_leakage']])
df_test['deal_or_not'] = leakage['deal_or_not_leakage']

print(df_test)
df_test.to_csv('submission_prediction_by_leakage.csv', index=False)

# deal_or_not_TP = leakage['deal_or_not'] * leakage['deal_or_not_leakage']
# error = sum(leakage['deal_or_not']) - sum(deal_or_not_TP) + sum(leakage['deal_or_not_leakage']) - sum(deal_or_not_TP)
# print(1 - error / len(leakage))

# print(df_test.dtypes)
# print(leakage.dtypes)
# print(pd.merge(df_test, leakage[['order_id', 'deal_or_not_leakage']]))
