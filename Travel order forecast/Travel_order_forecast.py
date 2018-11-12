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
# print(df_airline)

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

df_order_1 = df_order.merge(df_group[['group_id', 'Begin_Date', 'days', 'Area', 'SubLine', 'price']], on='group_id')
df_order_1['Order_Date'] = df_order_1.order_date.apply(lambda x: Convert_Date(x))
df_order_1['Source_1'] = df_order_1.source_1.apply(lambda x: int(x[11:]))
df_order_1['Source_2'] = df_order_1.source_2.apply(lambda x: int(x[11:]))
df_order_1['Unit'] = df_order_1.unit.apply(lambda x: int(x[11:]))
df_order_1['PreDays'] = (df_order_1['Begin_Date'] - df_order_1['Order_Date']).dt.days
df_order_1['Begin_Date_Weekday'] = df_order_1['Begin_Date'].dt.dayofweek
df_order_1['Order_Date_Weekday'] = df_order_1['Order_Date'].dt.dayofweek
df_order_1['Return_Date_Weekday'] = (df_order_1['Begin_Date'].dt.dayofweek + df_order_1['days']) % 7
df_order_1['Begin_Date_Month'] = df_order_1['Begin_Date'].dt.month
df_order_1['Order_Date_Month'] = df_order_1['Order_Date'].dt.month

df_order_2 = df_order_1[
    ['order_id', 'group_id', 'Order_Date_Month', 'Source_1', 'Source_2', 'Unit', 'people_amount', 'Begin_Date_Month', 'days',
     'Area', 'SubLine', 'price', 'PreDays', 'Begin_Date_Weekday', 'Order_Date_Weekday', 'Return_Date_Weekday']]
print(df_order_2)
df_train_1 = df_train.merge(df_order_2, on='order_id')
df_test_1 = df_test.merge(df_order_2, on='order_id')

tf_x = tf.placeholder(tf.float32, [None, 16])
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
for step in range(200):
    print(step)
    df_train_deal = df_train_1[df_train_1['deal_or_not'] == 1]
    df_train_not_deal = df_train_1[df_train_1['deal_or_not'] == 0]
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

test_deal_or_not = sess.run(tf_deal_or_not, {tf_x: df_test_1.drop(['deal_or_not'], 1)})
df_test['deal_or_not'] = test_deal_or_not
# print(df_test)
print(df_test['deal_or_not'].mean())

df_test.to_csv('submission.csv', index=False)
