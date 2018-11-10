import tensorflow as tf
import numpy as np
import pandas as pd

# df_test = pd.read_csv('cmp4_sample_submission.csv')
df_train = pd.read_csv("training-set.csv")
df_test = pd.read_csv("testing-set.csv")

tf_x = tf.placeholder(tf.float32, [None, 1])
tf_y = tf.placeholder(tf.float32, [None, 2])

tf_layer1 = tf.layers.dense(tf_x, 8, tf.nn.relu)
tf_layer2 = tf.layers.dense(tf_layer1, 2, tf.nn.relu)
tf_output = tf.layers.dense(tf_layer2, 2)

tf_deal_or_not = tf.argmax(tf_output, 1)

tf_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf_output, 1), tf.argmax(tf_y, 1)), 'float'))

tf_loss = tf.nn.softmax_cross_entropy_with_logits(logits=tf_output, labels=tf_y)
optimizer = tf.train.AdamOptimizer(0.5).minimize(tf_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_size = 100
for step in range(10):
    df_train_deal = df_train[df_train['deal_or_not'] == 1]
    df_train_not_deal = df_train[df_train['deal_or_not'] == 0]
    undersampling = df_train_deal.append(df_train_not_deal.sample(len(df_train_deal)))

    train_sample = undersampling.sample(frac=0.8)
    train_x = train_sample['order_id'][:, np.newaxis]
    train_y = np.eye(2)[train_sample['deal_or_not']]

    valid_sample = undersampling.sample(frac=0.2)
    valid_x = valid_sample['order_id'][:, np.newaxis]
    valid_y = np.eye(2)[valid_sample['deal_or_not']]

    print(len(train_sample))
    for i in range(0, len(train_sample), batch_size):
        x_batch = train_x[i:i + batch_size]
        y_batch = train_y[i:i + batch_size]
        # print(x_batch)
        # print(y_batch)
        accuracy, loss, _ = sess.run([tf_accuracy, tf_loss, optimizer], {tf_x: x_batch, tf_y: y_batch})
        # print(accuracy)
        # print(loss)

    print('train_accuracy{}'.format(sess.run(tf_accuracy, {tf_x: train_x, tf_y: train_y})))
    print('valid_accuracy{}'.format(sess.run(tf_accuracy, {tf_x: valid_x, tf_y: valid_y})))

test_deal_or_not = sess.run(tf_deal_or_not, {tf_x: df_test['order_id'][:, np.newaxis]})
df_test['deal_or_not'] = test_deal_or_not
# print(df_test)
print(df_test['deal_or_not'].mean())

df_test.to_csv('submission.csv', index=False)
