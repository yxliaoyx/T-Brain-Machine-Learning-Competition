import pandas as pd

df_test = pd.read_csv('testing-set.csv')
df_train = pd.read_csv('training-set.csv')
df_order = pd.read_csv('dataset/order.csv')
df_group = pd.read_csv('dataset/group.csv')
df_airline = pd.read_csv('dataset/airline.csv')
df_day_schedule = pd.read_csv('dataset/day_schedule.csv')

df_airline = df_airline[(df_airline['group_id'] != 1303) & (df_airline['group_id'] != 47252)]

df_airline_size = df_airline.groupby('group_id').size().reset_index().rename(columns={0: 'size'})
df_group_airline_size = pd.merge(df_group, df_airline_size, 'left').fillna(2)
# df_group_airline_size = pd.merge(df_group, df_airline_groupby[['group_id', 'size']], 'left').fillna(2)
print(df_group_airline_size['size'].max())
# print(df_order.merge(df_airline_transfer, on='group_id'))


# print(df_airline_transfer[df_airline_transfer == 7])
# print(df_airline_transfer.max())