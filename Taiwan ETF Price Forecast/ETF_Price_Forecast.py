import tensorflow as tf
import pandas as pd
import numpy as np

csv = pd.read_csv('tetfp.csv', encoding='big5')
data = csv[['開盤價(元)', '最高價(元)', '最低價(元)', '收盤價(元)', '成交張數(張)']][csv['代碼'] == 56]

x_data = data.drop(data.index[-5:])
y_data = data.drop(data.index[:4])['收盤價(元)']

tf_x = tf.placeholder(tf.float32, [5, 5])
tf_y = tf.placeholder(tf.float32, [6, 1])

layer1 = tf.layers.dense(tf_x, 5, tf.nn.relu)
layer2 = tf.layers.dense(layer1, 4, tf.nn.relu)
layer3 = tf.layers.dense(layer2, 3, tf.nn.relu)
layer4 = tf.layers.dense(layer3, 2, tf.nn.relu)
output = tf.layers.dense(layer4, 1)

up_down_prediction = tf.equal(tf.greater(tf_y[1:, :], tf_y[:-1, :]),
                              tf.greater(output, tf.concat([tf_y[:1, :], output[:-1, :]], 0)))

cost = tf.matmul([[.01, .015, .02, .025, .03]], tf.cast(up_down_prediction, tf.float32))

tf_loss = tf.losses.absolute_difference(tf_y[1:, :], output) + cost
optimizer = tf.train.AdamOptimizer(0.01).minimize(tf_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(len(x_data.index) - 5, -1, -1):
    # for i in range(1):
    x_batch = x_data[i:i + 5]
    # print('x_batch')
    # print(x_batch)
    y_batch = y_data[i:i + 6, np.newaxis]
    # print('y_batch')
    # print(y_batch)
    prediction, loss, _ = sess.run([output, tf_loss, optimizer], {tf_x: x_batch, tf_y: y_batch})
    # print('prediction')
    # print(prediction)
    # print('loss')
    print(loss)

for i in range(len(x_data.index) - 4):
    # for i in range(1):
    x_batch = x_data[i:i + 5]
    # print('x_batch')
    # print(x_batch)
    y_batch = y_data[i:i + 6, np.newaxis]
    # print('y_batch')
    # print(y_batch)
    prediction, loss, _ = sess.run([output, tf_loss, optimizer], {tf_x: x_batch, tf_y: y_batch})
    # print('prediction')
    # print(prediction)
    # print('loss')
    print(loss)

print(data[-5:])
result = sess.run(output, {tf_x: data[-5:]})
print('result')
print(result)
print('up_down_prediction')
print(result[0:1, :] - y_data[-1:, np.newaxis])
print(result[1:2, :] - result[0:1, :])
print(result[2:3, :] - result[1:2, :])
print(result[3:4, :] - result[2:3, :])
print(result[4:5, :] - result[3:4, :])
