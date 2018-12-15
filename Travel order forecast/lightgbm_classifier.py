import lightgbm as lgb
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pylab as plt

df_order = pd.read_csv('df_order.csv', dtype={'order_id': str, 'group_id': str})
df_test = pd.read_csv('testing-set.csv', dtype={'order_id': str})
df_train = pd.read_csv('training-set.csv', dtype={'order_id': str})

for feature in ['product_name_in_brackets',
                'schedule_random_title_in_brackets',
                'schedule_random_city',
                'src_random_airport_go',
                'src_random_airport_go_1st_letter',
                'dst_random_airport_go_1st_letter',
                'src_random_airport_go_2nd_letter',
                'dst_random_airport_go_2nd_letter',
                'src_random_airport_go_3rd_letter',
                'dst_random_airport_go_3rd_letter',
                'dst_random_airport_go',
                'src_random_airport_back',
                'dst_random_airport_back',
                'src_random_airport_back_1st_letter',
                'dst_random_airport_back_1st_letter',
                'src_random_airport_back_2nd_letter',
                'dst_random_airport_back_2nd_letter',
                'src_random_airport_back_3rd_letter',
                'dst_random_airport_back_3rd_letter']:
    le = LabelEncoder()
    le.fit(df_order[feature].astype(str))
    df_order[feature] = le.transform(df_order[feature].astype(str))

features = ['source_1',
            'source_2',
            'unit',
            'people_amount',
            'sub_line',
            'area',
            'days',
            'price',
            'product_name_in_brackets',
            'product_name_in_brackets_len',
            'product_name_brackets_sum',
            'product_name_brackets_diff',
            'product_name_len',
            'promotion_prog_len',
            'schedule_random_day',
            'schedule_random_title_len',
            'schedule_random_title_in_brackets',
            'schedule_random_city',
            'schedule_day_sum',
            'schedule_title_len_sum',
            'src_random_airport_go',
            'dst_random_airport_go',
            'src_random_airport_go_1st_letter',
            'dst_random_airport_go_1st_letter',
            'src_random_airport_go_2nd_letter',
            'dst_random_airport_go_2nd_letter',
            'src_random_airport_go_3rd_letter',
            'dst_random_airport_go_3rd_letter',
            'src_random_airport_back',
            'dst_random_airport_back',
            'src_random_airport_back_1st_letter',
            'dst_random_airport_back_1st_letter',
            'src_random_airport_back_2nd_letter',
            'dst_random_airport_back_2nd_letter',
            'src_random_airport_back_3rd_letter',
            'dst_random_airport_back_3rd_letter',
            'transfer',
            'source_1 + source_2',
            'source_1 + unit',
            'source_2 + unit',
            'source_1 * source_2',
            'source_1 * unit',
            'source_2 * unit',
            'predays',
            'begin_date_weekday',
            'order_date_weekday',
            'return_date_weekday',
            'order_date_year',
            'begin_date_year',
            'order_date_month',
            'begin_date_month',
            'order_date_day',
            'begin_date_day',
            'order_date_week',
            'begin_date_week',
            'order_date_quarter',
            'begin_date_quarter',
            'price // predays',
            'predays // days',
            'price // days']

df_test = pd.merge(df_test, df_order, 'left')
test_x = df_test[features]

df_train = pd.merge(df_train, df_order, 'left')
df_train_deal = df_train[df_train['deal_or_not'] == 1]
df_train_not_deal = df_train[df_train['deal_or_not'] == 0]

undersampling = df_train_deal.append(df_train_not_deal.sample(len(df_train_deal)))
train_x = undersampling[features]
train_y = undersampling['deal_or_not']
# train_x = df_train[features]
# train_y = df_train['deal_or_not']

X_train, X_valid, Y_train, Y_valid = train_test_split(
    train_x,
    train_y,
    test_size=0.2,
    random_state=1,
    stratify=train_y
)

lgb_train = lgb.Dataset(X_train, Y_train)
lgb_eval = lgb.Dataset(X_valid, Y_valid, reference=lgb_train)

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss', 'auc'},
    'learning_rate': 0.01,
    'max_depth': 10,
    'num_leaves': 1000,
    # 'min_sum_hessian_in_leaf': 0.01,
    # 'min_data_in_leaf': 100,
    'bagging_fraction': 0.8,
    'feature_fraction': 0.5,
    'lambda_l1': 0.001,
    # 'lambda_l2': 0.001
    # 'min_gain_to_split': 0.2
    'verbose': 2,
    # 'is_unbalance': True
}

gbm = lgb.train(params,
                lgb_train,
                valid_sets=lgb_eval,
                num_boost_round=5000,
                early_stopping_rounds=100)

preds = gbm.predict(test_x, num_iteration=gbm.best_iteration)
df_test['deal_or_not'] = preds
print(df_test['deal_or_not'].describe())
df_test.to_csv('submission.csv', columns=['order_id', 'deal_or_not'], index=False)

threshold = 0.5
for pred in preds:
    result = 1 if pred > threshold else 0

importance = gbm.feature_importance()
names = gbm.feature_name()
with open('./feature_importance.txt', 'w+') as file:
    for index, im in enumerate(importance):
        string = names[index] + ', ' + str(im) + '\n'
        file.write(string)

lgb.plot_importance(gbm, figsize=(16, 9))
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()
