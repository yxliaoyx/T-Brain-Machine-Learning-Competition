import lightgbm as lgb
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

print("Loading Data ... ")
df_train_1 = pd.read_csv('df_train_1.csv')
df_test = pd.read_csv('testing-set.csv')

# 导入数据
# train_x, train_y, test_x = load_data()

df_train_deal = df_train_1[df_train_1['deal_or_not'] == 1]
df_train_not_deal = df_train_1[df_train_1['deal_or_not'] == 0]
undersampling = df_train_deal.append(df_train_not_deal.sample(len(df_train_deal)))
train_x = undersampling.drop(['deal_or_not'], 1)
train_y = undersampling['deal_or_not']
test_x = df_test.drop(['deal_or_not'], 1)

X_train, X_test, y_train, y_test = train_test_split(
    train_x,
    train_y,
    test_size=0.2,
    random_state=1,
    stratify=train_y  # 这里保证分割后y的比例分布与原数据一致
)

# X_train = X
# y_train = y
# X_test = val_X
# y_test = val_y

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss', 'auc'},  # 二进制对数损失
    'num_leaves': 5,
    'max_depth': 6,
    'min_data_in_leaf': 450,
    'learning_rate': 0.1,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.95,
    'bagging_freq': 5,
    'lambda_l1': 1,
    'lambda_l2': 0.001,  # 越小l2正则程度越高
    'min_gain_to_split': 0.2,
    'verbose': 5,
    'is_unbalance': True
}

# train
print('Start training...')
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10000,
                valid_sets=lgb_eval,
                early_stopping_rounds=500)

print('Start predicting...')

preds = gbm.predict(test_x, num_iteration=gbm.best_iteration)  # 输出的是概率结果
df_test['deal_or_not'] = preds
print(df_test['deal_or_not'].mean())
df_test.to_csv('submission.csv', index=False)

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
