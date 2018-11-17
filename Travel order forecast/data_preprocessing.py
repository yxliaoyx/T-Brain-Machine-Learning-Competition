import pandas as pd

df_test = pd.read_csv('testing-set.csv')
df_train = pd.read_csv('training-set.csv')
df_order = pd.read_csv('dataset/order.csv')
df_group = pd.read_csv('dataset/group.csv')
df_airline = pd.read_csv('dataset/airline.csv')
df_day_schedule = pd.read_csv('dataset/day_schedule.csv')

df_airline = df_airline[(df_airline['group_id'] != 1303) & (df_airline['group_id'] != 47252)]

df_airline_transfer = df_airline.groupby('group_id').size().reset_index().rename(columns={0: 'transfer'})
df_group_airline_transfer  = pd.merge(df_group, df_airline_transfer , 'left').fillna(2)

df_day_schedule['title_len'] = df_day_schedule['title'].apply(lambda x: len(x))
df_day_schedule_sum = df_day_schedule.groupby('group_id').sum().reset_index()
print(df_day_schedule_sum['title_len'].max())