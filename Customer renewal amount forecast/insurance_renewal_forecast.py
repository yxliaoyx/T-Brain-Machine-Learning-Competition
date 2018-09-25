import tensorflow as tf
import pandas as pd
import numpy as np

df_test_set = pd.read_csv('data/testing-set.csv')

# df_data_train_set = pd.read_csv('data/training-set.csv')
# df_duplicate_policy = pd.read_csv('data/duplicate_policy.csv')
# df_duplicate_policy_train = pd.merge(df_duplicate_policy, df_train_set)
# df_duplicate_policy_train['Next_Premium'] = df_duplicate_policy_train['Next_Premium'].apply(lambda x: x / 2)
# df_duplicate_policy_train.to_csv('duplicate_policy_train.csv', index=False)
# df_duplicate_policy_train = pd.read_csv('duplicate_policy_train.csv')
# pd.merge(df_train_set, df_duplicate_policy_train, 'left', 'Policy_Number').fillna(0).to_csv('Next_Premium_difference.csv', index=False)
# df_Next_Premium_difference = pd.read_csv('Next_Premium_difference.csv')
# df_data_train_set['Next_Premium'] = df_Next_Premium_difference['Next_Premium_x'] - df_Next_Premium_difference['Next_Premium_y']
# df_data_train_set.to_csv('train_set.csv', index=False)
df_train_set = pd.read_csv('train_set.csv')

df_claim = pd.read_csv('data/policy_claim/claim_0702.csv')[
    ['Nature_of_the_claim', 'Policy_Number', "Driver's_Gender", "Driver's_Relationship_with_Insured", 'DOB_of_Driver',
     'Marital_Status_of_Driver', 'Accident_Date', 'Paid_Loss_Amount', 'paid_Expenses_Amount', 'Salvage_or_Subrogation?',
     'At_Fault?', 'Claim_Status_(close,_open,_reopen_etc)', 'Deductible', 'number_of_claimants', 'Accident_Time']]
# df_claim['DOB_of_Driver'] = df_claim['DOB_of_Driver'].str.extract('(19..)', expand=True).fillna(2018)
# df_claim['DOB_of_Driver'] = df_claim['DOB_of_Driver'].apply(lambda x: 2018 - int(x))
# df_claim['Accident_Date'] = df_claim['Accident_Date'].str.extract('(..$)', expand=True)
# df_claim['Accident_Time'] = df_claim['Accident_Time'].str.extract('(^..)', expand=True)
# df_claim.groupby('Policy_Number').mean().to_csv('claim_mean.csv')
df_claim_mean = pd.read_csv('claim_mean.csv')

df_policy = pd.read_csv('data/policy_claim/policy_0702.csv')
# df_policy_feature = df_policy[[
#     'Policy_Number', 'Cancellation', 'Manafactured_Year_and_Month', 'Engine_Displacement_(Cubic_Centimeter)',
#     'Imported_or_Domestic_Car', 'qpt', 'Main_Insurance_Coverage_Group', 'Insured_Amount1', 'Insured_Amount2',
#     'Insured_Amount3', 'Coverage_Deductible_if_applied', 'Premium', 'Replacement_cost_of_insured_vehicle',
#     'Multiple_Products_with_TmNewa_(Yes_or_No?)', 'lia_class', 'plia_acc', 'pdmg_acc', 'fassured', 'ibirth', 'fsex',
#     'fmarriage', 'dbirth']]
#
# df_policy_feature['Cancellation'] = df_policy_feature['Cancellation'].apply({'Y': 0, ' ': 1}.get)
# df_policy_feature['Manafactured_Year_and_Month'] = df_policy_feature['Manafactured_Year_and_Month'].apply(
#     lambda x: 2018 - int(x))
# df_policy_feature['Main_Insurance_Coverage_Group'] = df_policy_feature['Main_Insurance_Coverage_Group'].apply(
#     {'車責': 0, '竊盜': 1, '車損': 2}.get)
# df_policy_feature['ibirth'] = df_policy_feature['ibirth'].str.extract('(19..)', expand=True).fillna(2018)
# df_policy_feature['ibirth'] = df_policy_feature['ibirth'].apply(lambda x: 2018 - int(x))
# df_policy_feature['fsex'] = df_policy_feature['fsex'].apply({'1': 1, '2': 2, ' ': 0}.get).fillna(0)
# df_policy_feature['fmarriage'] = df_policy_feature['fmarriage'].apply({'1': 1, '2': 2, ' ': 0}.get).fillna(0)
# df_policy_feature['dbirth'] = df_policy_feature['dbirth'].str.extract('(19..)', expand=True).fillna(2018)
# df_policy_feature['dbirth'] = df_policy_feature['dbirth'].apply(lambda x: 2018 - int(x))
# df_policy_feature.to_csv('policy_feature.csv', index=False)
df_policy_feature = pd.read_csv('policy_feature.csv')

# df_policy['Prior_Policy_Number'] = df_policy['Prior_Policy_Number'].fillna(0)
# df_policy[['Policy_Number', 'Prior_Policy_Number', 'Premium']].groupby(['Policy_Number', 'Prior_Policy_Number']).sum().to_csv('Premium_Prior_sum.csv')
df_Premium_Prior_sum = pd.read_csv('Premium_Prior_sum.csv')
df_Premium_sum = df_Premium_Prior_sum[['Policy_Number', 'Premium']].rename(columns={'Premium': 'Premium_sum'})
df_Prior_sum = df_Premium_Prior_sum[['Prior_Policy_Number', 'Premium']][
    df_Premium_Prior_sum['Prior_Policy_Number'] != '0'].rename(
    columns={'Prior_Policy_Number': 'Policy_Number', 'Premium': 'Premium_Next'})

df_policy_feature.groupby('Policy_Number').mean().rename(columns={'Premium': 'Premium_mean'}).to_csv('policy_mean.csv')
df_policy_mean = pd.read_csv('policy_mean.csv')

df_policy_mean = pd.merge(df_policy_mean, df_policy_feature.groupby('Policy_Number').size().reset_index(),
                          'left').fillna(0).rename(columns={0: 'policies'})
df_Policy_claims = pd.merge(df_policy_mean, df_claim.groupby('Policy_Number').size().reset_index(),
                            'left').fillna(0).rename(columns={0: 'claims'})

df_Policy_claims = pd.merge(df_Policy_claims, df_claim_mean, 'left').fillna(0)

df_train_Premium_sum = pd.merge(df_train_set, df_Premium_sum)
df_test_Premium_sum = pd.merge(df_test_set, df_Premium_sum)

df_train_Premium_sum = pd.merge(df_train_Premium_sum, df_Prior_sum, 'left').fillna(0)
df_test_Premium_sum = pd.merge(df_test_Premium_sum, df_Prior_sum, 'left').fillna(0)

df_train = pd.merge(df_train_Premium_sum, df_Policy_claims)
df_test = pd.merge(df_test_Premium_sum, df_Policy_claims)

sample_train = df_train.sample(frac=0.8)
train_x = sample_train.drop(['Policy_Number', 'Next_Premium'], 1)
train_y = sample_train['Next_Premium']

sample_valid = df_train.sample(frac=0.2)
valid_x = sample_valid.drop(['Policy_Number', 'Next_Premium'], 1)
valid_y = sample_valid['Next_Premium']

tf_x = tf.placeholder(tf.float32, [None, 37])
tf_y = tf.placeholder(tf.float32, [None, 1])

tf_layer1 = tf.layers.dense(tf_x, 80, tf.nn.relu)
tf_layer2 = tf.layers.dense(tf_layer1, 20, tf.nn.relu)
tf_layer3 = tf.layers.dense(tf_layer2, 78, tf.nn.relu)
tf_layer4 = tf.layers.dense(tf_layer3, 22, tf.nn.relu)
tf_layer5 = tf.layers.dense(tf_layer4, 8, tf.nn.relu)
tf_layer6 = tf.layers.dense(tf_layer5, 7, tf.nn.relu)
tf_output = tf.layers.dense(tf_layer6, 1, tf.nn.relu)

tf_loss = tf.losses.absolute_difference(tf_y, tf_output)
tf_optimizer = tf.train.AdamOptimizer(0.00005).minimize(tf_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# batch_size = 1297
batch_size = 130

for step in range(2000):
    for i in range(0, len(sample_train), batch_size):
        sess.run(tf_optimizer, {tf_x: train_x[i:i + batch_size], tf_y: train_y[i:i + batch_size, np.newaxis]})

    print(sess.run([tf_loss], {tf_x: train_x, tf_y: train_y[:, np.newaxis]}))
    print(sess.run([tf_loss], {tf_x: valid_x, tf_y: valid_y[:, np.newaxis]}))
    print(step)

df_test_set['Next_Premium'] = sess.run(tf_output, {tf_x: df_test.drop(['Policy_Number', 'Next_Premium'], 1)})
df_test_set.to_csv('submission_test.csv', columns=['Policy_Number', 'Next_Premium'], index=False)

# df_test_Next_Premium = pd.merge(df_test_set.drop('Next_Premium', 1), df_Prior_sum)
# df_test_Next_Premium.to_csv('test_Next_Premium.csv', index=False)
df_test_Next_Premium = pd.read_csv('test_Next_Premium.csv')

for index, row in df_test_Next_Premium.iterrows():
    df_test_set['Next_Premium'][df_test_set['Policy_Number'] == row['Policy_Number']] = row['Premium_Next']
df_test_set.to_csv('submission.csv', columns=['Policy_Number', 'Next_Premium'], index=False)

for index, row in df_test_Next_Premium.iterrows():
    df_test_set['Next_Premium'][df_test_set['Policy_Number'] == row['Policy_Number']] = row['Premium_Next'] * 2
df_test_set.to_csv('submission_double.csv', columns=['Policy_Number', 'Next_Premium'], index=False)
