import lightgbm as lgb
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

print("Loading Data ... ")
df_order = pd.read_csv('df_order.csv')
df_test = pd.read_csv('testing-set.csv')
df_train = pd.read_csv('training-set.csv')

for feature in ['src_airport_go', 'dst_airport_go', 'src_airport_back', 'dst_airport_back']:
    le = LabelEncoder()
    le.fit(df_order[feature].astype(str))
    df_order[feature] = le.transform(df_order[feature].astype(str))

features = ['source_1', 'source_2', 'unit', 'people_amount', 'sub_line', 'area', 'days', 'price', 'product_name_len',
            'promotion_prog_len', 'random_day', 'random_title_len', 'day_sum', 'title_len_sum', 'src_airport_go',
            'dst_airport_go', 'src_airport_back', 'dst_airport_back', 'transfer', 'predays', 'begin_date_weekday',
            'order_date_weekday', 'return_date_weekday', 'order_date_year', 'begin_date_year', 'order_date_month',
            'begin_date_month', 'order_date_day', 'begin_date_day', 'order_date_week', 'begin_date_week',
            'order_date_quarter', 'begin_date_quarter']

df_test = pd.merge(df_test, df_order, 'left')
test_x = df_test[features]

df_train = pd.merge(df_train, df_order, 'left')
df_train_deal = df_train[df_train['deal_or_not'] == 1]
df_train_not_deal = df_train[df_train['deal_or_not'] == 0]
undersampling = df_train_deal.append(df_train_not_deal.sample(len(df_train_deal)))
train_x = undersampling[features]
train_y = undersampling['deal_or_not']

X_train, X_valid, Y_train, Y_valid = train_test_split(
    train_x,
    train_y,
    test_size=0.2,
    random_state=1,
    stratify=train_y
)

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, Y_train)
lgb_eval = lgb.Dataset(X_valid, Y_valid, reference=lgb_train)

# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss', 'auc'},
    'num_leaves': 100,
    'max_depth': 20,
    'min_data_in_leaf': 450,
    'learning_rate': 0.002,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.9,
    'bagging_freq': 5,
    'lambda_l1': 0.001,
    # 'lambda_l2': 0.001,
    'min_gain_to_split': 0.5,
    'verbose': 5,
    'is_unbalance': True
}

# train
print('Start training...')
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10000,
                valid_sets=lgb_eval,
                early_stopping_rounds=1000)

print('Start predicting...')
preds = gbm.predict(test_x, num_iteration=gbm.best_iteration)  # 输出的是概率结果
df_test['deal_or_not'] = preds
print(df_test['deal_or_not'].mean())
df_test.to_csv('submission.csv', columns=['order_id', 'deal_or_not'], index=False)

# 导出结果
threshold = 0.5
for pred in preds:
    result = 1 if pred > threshold else 0

# 导出特征重要性
importance = gbm.feature_importance()
names = gbm.feature_name()
with open('./feature_importance.txt', 'w+') as file:
    for index, im in enumerate(importance):
        string = names[index] + ', ' + str(im) + '\n'
        file.write(string)
