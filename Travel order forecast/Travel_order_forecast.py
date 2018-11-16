import tensorflow as tf
import numpy as np
import pandas as pd

df_test = pd.read_csv('testing-set.csv')
df_train = pd.read_csv('training-set.csv')
df_order = pd.read_csv('dataset/order.csv')
df_group = pd.read_csv('dataset/group.csv')
df_airline = pd.read_csv('dataset/airline.csv')
df_day_schedule = pd.read_csv('dataset/day_schedule.csv')

df_airline = df_airline[(df_airline['group_id'] != 1303) & (df_airline['group_id'] != 47252)]

month = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
         'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}


def Convert_Date(x):
    Year = '20' + x[-2:]
    Month = month[x[-6:-3]]
    Day = x[:-7]
    return pd.to_datetime(Year + '-' + Month + '-' + Day)


df_group['Begin_Date'] = df_group.begin_date.apply(lambda x: Convert_Date(x))
df_group['SubLine'] = df_group.sub_line.apply(lambda x: int(x[14:]))
df_group['Area'] = df_group.area.apply(lambda x: int(x[11:]))

df_airline_size = df_airline.groupby('group_id').size().reset_index().rename(columns={0: 'size'})
df_group_airline_size = pd.merge(df_group, df_airline_size, 'left').fillna(2)

df_order_1 = df_order.merge(
    df_group_airline_size[['group_id', 'Begin_Date', 'days', 'Area', 'SubLine', 'price', 'size']], on='group_id')
df_order_1['Order_Date'] = df_order_1.order_date.apply(lambda x: Convert_Date(x))
df_order_1['Source_1'] = df_order_1.source_1.apply(lambda x: int(x[11:]))
df_order_1['Source_2'] = df_order_1.source_2.apply(lambda x: int(x[11:]))
df_order_1['Unit'] = df_order_1.unit.apply(lambda x: int(x[11:]))
df_order_1['PreDays'] = (df_order_1['Begin_Date'] - df_order_1['Order_Date']).dt.days
df_order_1['Begin_Date_Weekday'] = df_order_1['Begin_Date'].dt.dayofweek
df_order_1['Order_Date_Weekday'] = df_order_1['Order_Date'].dt.dayofweek
df_order_1['Return_Date_Weekday'] = (df_order_1['Begin_Date'].dt.dayofweek + df_order_1['days']) % 7
df_order_1['Order_Date_Year'] = df_order_1['Order_Date'].dt.year % 100
df_order_1['Begin_Date_Year'] = df_order_1['Begin_Date'].dt.year % 100
df_order_1['Order_Date_Month'] = df_order_1['Order_Date'].dt.month
df_order_1['Begin_Date_Month'] = df_order_1['Begin_Date'].dt.month
df_order_1['Order_Date_Day'] = df_order_1['Order_Date'].dt.day
df_order_1['Begin_Date_Day'] = df_order_1['Begin_Date'].dt.day
df_order_1['Order_Date_Week'] = df_order_1['Order_Date'].dt.week
df_order_1['Begin_Date_Week'] = df_order_1['Begin_Date'].dt.week
df_order_1['Order_Date_Quarter'] = df_order_1['Order_Date'].dt.quarter
df_order_1['Begin_Date_Quarter'] = df_order_1['Begin_Date'].dt.quarter

# print(df_order_1)
df_order_2 = df_order_1[['order_id', 'size', 'PreDays']]
# df_order_2 = df_order_1[
#     ['order_id', 'group_id', 'Source_1', 'Source_2', 'Unit', 'people_amount', 'days', 'Area', 'SubLine', 'price',
#      'size', 'PreDays', 'Begin_Date_Weekday', 'Order_Date_Weekday', 'Return_Date_Weekday', 'Order_Date_Year',
#      'Begin_Date_Year', 'Order_Date_Month', 'Begin_Date_Month', 'Order_Date_Day', 'Begin_Date_Day', 'Order_Date_Week',
#      'Begin_Date_Week', 'Order_Date_Quarter', 'Begin_Date_Quarter']]
# print(df_order_2)

df_train_1 = df_train.merge(df_order_2, on='order_id')
df_test_1 = df_test.merge(df_order_2, on='order_id')
# df_train_1.to_csv('df_train_1.csv', index=False)
df_train_deal = df_train_1[df_train_1['deal_or_not'] == 1]
df_train_not_deal = df_train_1[df_train_1['deal_or_not'] == 0]
# print(df_train_deal['order_id'].min())
# print(df_train_deal['order_id'].max())
# print(df_train_not_deal['order_id'].min())
# print(df_train_not_deal['order_id'].max())

'''
df_train_1['x'] = 1
# df_train_1['x'] = np.random.randint(2,size=len(df_train_1))
df_train_1[df_train_1['order_id'] > df_train_deal['order_id'].max()] = 0
print(df_train_1)
df_train_1['acc'] = df_train_1['x'] == df_train_1['deal_or_not']
print(df_train_1['acc'].mean())
'''

# df_train_1['x'] = df_train_1['order_id135'] * df_train_1['deal_or_not']
# x = len(df_train_1[df_train_1['x'] == 0])
# PreDays = len(df_train_1[df_train_1['order_id135'] == 0])
# deal_or_not = len(df_train_1[df_train_1['deal_or_not'] == 0])
# print(2 * x - PreDays - deal_or_not)
# print((2 * x - PreDays - deal_or_not) / len(df_train_1))

tf_x = tf.placeholder(tf.float32, [None, 3])
tf_y = tf.placeholder(tf.float32, [None, 2])

tf_layer1 = tf.layers.dense(tf_x, 80, tf.nn.relu)
tf_layer2 = tf.layers.dense(tf_layer1, 20, tf.nn.relu)
tf_layer3 = tf.layers.dense(tf_layer2, 78, tf.nn.relu)
tf_layer4 = tf.layers.dense(tf_layer3, 22, tf.nn.relu)
tf_layer5 = tf.layers.dense(tf_layer4, 8, tf.nn.relu)
tf_layer6 = tf.layers.dense(tf_layer5, 7, tf.nn.relu)
tf_output = tf.layers.dense(tf_layer6, 2)

tf_deal_or_not = tf.argmax(tf_output, 1)

tf_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf_output, 1), tf.argmax(tf_y, 1)), 'float'))

tf_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf_output, labels=tf_y))
optimizer = tf.train.AdamOptimizer(0.000005).minimize(tf_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_size = 100
for step in range(100):
    print(step)
    df_train_deal = df_train_1[df_train_1['deal_or_not'] == 1]
    df_train_not_deal = df_train_1[df_train_1['deal_or_not'] == 0]
    # print(df_train_deal.mean())
    # print(df_train_not_deal.mean())
    undersampling = df_train_deal.append(df_train_not_deal.sample(len(df_train_deal)))

    train_sample = undersampling.sample(frac=0.8)
    train_x = train_sample.drop(['deal_or_not'], 1)
    train_y = np.eye(2)[train_sample['deal_or_not']]

    valid_sample = undersampling.sample(frac=0.2)
    valid_x = valid_sample.drop(['deal_or_not'], 1)
    valid_y = np.eye(2)[valid_sample['deal_or_not']]

    for i in range(0, len(train_sample), batch_size):
        x_batch = train_x[i:i + batch_size]
        y_batch = train_y[i:i + batch_size]
        # print(x_batch)
        # print(y_batch)
        accuracy, loss, _ = sess.run([tf_accuracy, tf_loss, optimizer], {tf_x: x_batch, tf_y: y_batch})
        # print(accuracy)
        # print(loss)

    print('train {}'.format(sess.run([tf_accuracy, tf_loss], {tf_x: train_x, tf_y: train_y})))
    print('valid {}'.format(sess.run([tf_accuracy, tf_loss], {tf_x: valid_x, tf_y: valid_y})))

print(sess.run(tf_accuracy, {tf_x: df_train_1.drop(['deal_or_not'], 1), tf_y: np.eye(2)[df_train_1['deal_or_not']]}))

test_deal_or_not = sess.run(tf_deal_or_not, {tf_x: df_test_1.drop(['deal_or_not'], 1)})
df_test['deal_or_not'] = test_deal_or_not
# print(df_test)
print(df_test['deal_or_not'].mean())

df_test.to_csv('submission.csv', index=False)
