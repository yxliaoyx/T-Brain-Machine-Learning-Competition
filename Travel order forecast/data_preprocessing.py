import pandas as pd
import numpy as np

df_order = pd.read_csv('dataset/order.csv', dtype={'order_id': str, 'group_id': str})
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


df_group['begin_date'] = df_group['begin_date'].apply(lambda x: Convert_Date(x))
df_group['sub_line'] = df_group['sub_line'].apply(lambda x: int(x[14:]))
df_group['area'] = df_group['area'].apply(lambda x: int(x[11:]))
df_group['product_name_in_brackets'] = df_group['product_name'].str.extract('[《【]?(.+?)[》】]', expand=True).fillna('')
df_group['product_name_in_brackets_len'] = df_group['product_name_in_brackets'].apply(lambda x: len(str(x)))
left_bracket_count = df_group['product_name'].str.count('《')
right_bracket_count = df_group['product_name'].str.count('》')
df_group['product_name_brackets_sum'] = left_bracket_count + right_bracket_count
df_group['product_name_brackets_diff'] = left_bracket_count - right_bracket_count
df_group['product_name_len'] = df_group['product_name'].apply(lambda x: len(str(x)))
df_group['promotion_prog_len'] = df_group['promotion_prog'].apply(lambda x: len(str(x)))

df_day_schedule['schedule_title_len'] = df_day_schedule['title'].apply(lambda x: len(str(x)))
df_day_schedule['schedule_title_in_brackets'] = df_day_schedule['title'].str.extract('【(.+?)】', expand=True).fillna('')

df_day_schedule_random = df_day_schedule.groupby('group_id').agg(np.random.choice).reset_index()
df_day_schedule_random['schedule_title_len'] = df_day_schedule_random['title'].apply(lambda x: len(str(x)))
df_day_schedule_random['schedule_city'] = df_day_schedule_random['title'].apply(lambda x: str(x)[:2])
df_day_schedule_random = df_day_schedule_random.rename(
    columns={'day': 'schedule_random_day', 'schedule_title_len': 'schedule_random_title_len',
             'schedule_title_in_brackets': 'schedule_random_title_in_brackets',
             'schedule_city': 'schedule_random_city'})
df_group = pd.merge(df_group, df_day_schedule_random[
    ['group_id', 'schedule_random_day', 'schedule_random_title_len', 'schedule_random_title_in_brackets',
     'schedule_random_city']], 'left')

df_day_schedule_sum = df_day_schedule.groupby('group_id').sum().reset_index()
df_day_schedule_sum = df_day_schedule_sum.rename(
    columns={'day': 'schedule_day_sum', 'schedule_title_len': 'schedule_title_len_sum'})
df_group = pd.merge(df_group, df_day_schedule_sum[['group_id', 'schedule_day_sum', 'schedule_title_len_sum']], 'left')

df_airport_go = df_airline[['group_id', 'src_airport', 'dst_airport']][df_airline['go_back'] == '去程']
df_random_airport_go = df_airport_go.groupby('group_id').agg(np.random.choice).reset_index()
df_airport_back = df_airline[['group_id', 'src_airport', 'dst_airport']][df_airline['go_back'] == '回程']
df_random_airport_back = df_airport_back.groupby('group_id').agg(np.random.choice).reset_index()

df_group = pd.merge(df_group, df_random_airport_go, 'left')
df_group = df_group.rename(columns={'src_airport': 'src_random_airport_go', 'dst_airport': 'dst_random_airport_go'})
df_group['src_random_airport_go_1st_letter'] = df_group['src_random_airport_go'].apply(lambda x: str(x)[0])
df_group['dst_random_airport_go_1st_letter'] = df_group['dst_random_airport_go'].apply(lambda x: str(x)[0])
df_group['src_random_airport_go_2nd_letter'] = df_group['src_random_airport_go'].apply(lambda x: str(x)[1])
df_group['dst_random_airport_go_2nd_letter'] = df_group['dst_random_airport_go'].apply(lambda x: str(x)[1])
df_group['src_random_airport_go_3rd_letter'] = df_group['src_random_airport_go'].apply(lambda x: str(x)[2])
df_group['dst_random_airport_go_3rd_letter'] = df_group['dst_random_airport_go'].apply(lambda x: str(x)[2])
df_group = pd.merge(df_group, df_random_airport_back, 'left')
df_group = df_group.rename(columns={'src_airport': 'src_random_airport_back', 'dst_airport': 'dst_random_airport_back'})
df_group['src_random_airport_back_1st_letter'] = df_group['src_random_airport_back'].apply(lambda x: str(x)[0])
df_group['dst_random_airport_back_1st_letter'] = df_group['dst_random_airport_back'].apply(lambda x: str(x)[0])
df_group['src_random_airport_back_2nd_letter'] = df_group['src_random_airport_back'].apply(lambda x: str(x)[1])
df_group['dst_random_airport_back_2nd_letter'] = df_group['dst_random_airport_back'].apply(lambda x: str(x)[1])
df_group['src_random_airport_back_3rd_letter'] = df_group['src_random_airport_back'].apply(lambda x: str(x)[2])
df_group['dst_random_airport_back_3rd_letter'] = df_group['dst_random_airport_back'].apply(lambda x: str(x)[2])

df_airline_transfer = df_airline.groupby('group_id').size().reset_index().rename(columns={0: 'transfer'}).fillna(2)
df_group = pd.merge(df_group, df_airline_transfer, 'left')

df_order = pd.merge(df_order, df_group, 'left')

df_order['order_date'] = df_order['order_date'].apply(lambda x: Convert_Date(x))
df_order['source_1'] = df_order['source_1'].apply(lambda x: int(x[-1:]))
df_order['source_2'] = df_order['source_2'].apply(lambda x: int(x[-1:]))
df_order['unit'] = df_order['unit'].apply(lambda x: int(x[-1:]))
df_order['source_1 + source_2'] = df_order['source_1'] + df_order['source_2']
df_order['source_1 + unit'] = df_order['source_1'] + df_order['unit']
df_order['source_2 + unit'] = df_order['source_2'] + df_order['unit']
df_order['source_1 * source_2'] = df_order['source_1'] * df_order['source_2']
df_order['source_1 * unit'] = df_order['source_1'] * df_order['unit']
df_order['source_2 * unit'] = df_order['source_2'] * df_order['unit']
df_order['predays'] = (df_order['begin_date'] - df_order['order_date']).dt.days
df_order['begin_date_weekday'] = df_order['begin_date'].dt.dayofweek
df_order['order_date_weekday'] = df_order['order_date'].dt.dayofweek
df_order['return_date_weekday'] = (df_order['begin_date'].dt.dayofweek + df_order['days']) % 7
df_order['order_date_year'] = df_order['order_date'].dt.year
df_order['begin_date_year'] = df_order['begin_date'].dt.year
df_order['order_date_month'] = df_order['order_date'].dt.month
df_order['begin_date_month'] = df_order['begin_date'].dt.month
df_order['order_date_day'] = df_order['order_date'].dt.day
df_order['begin_date_day'] = df_order['begin_date'].dt.day
df_order['order_date_week'] = df_order['order_date'].dt.week
df_order['begin_date_week'] = df_order['begin_date'].dt.week
df_order['order_date_quarter'] = df_order['order_date'].dt.quarter
df_order['begin_date_quarter'] = df_order['begin_date'].dt.quarter

df_order['price // predays'] = df_order['price'] // df_order['predays']
df_order['predays // days'] = df_order['predays'] // df_order['days']
df_order['price // days'] = df_order['price'] // df_order['days']

print(df_order.describe())

df_order.to_csv('df_order.csv', index=False)
