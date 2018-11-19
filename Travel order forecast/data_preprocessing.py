import pandas as pd

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


df_group['Begin_Date'] = df_group['begin_date'].apply(lambda x: Convert_Date(x))
df_group['SubLine'] = df_group['sub_line'].apply(lambda x: int(x[14:]))
df_group['Area'] = df_group['area'].apply(lambda x: int(x[11:]))

df_airport_go = df_airline[['group_id', 'src_airport', 'dst_airport']][df_airline['go_back'] == '去程']
df_airport_go = df_airport_go.groupby('group_id').sum().reset_index()
df_airport_back = df_airline[['group_id', 'src_airport', 'dst_airport']][df_airline['go_back'] == '回程']
df_airport_back = df_airport_back.groupby('group_id').sum().reset_index()

df_order = pd.merge(df_order, df_airport_go, 'left')
df_order = df_order.rename(columns={'src_airport': 'src_airport_go', 'dst_airport': 'dst_airport_go'})
df_order = pd.merge(df_order, df_airport_back, 'left')
df_order = df_order.rename(columns={'src_airport': 'src_airport_back', 'dst_airport': 'dst_airport_back'})

df_order = pd.merge(df_order, df_group[['group_id', 'Begin_Date', 'days', 'Area', 'SubLine', 'price']], 'left')

df_order['Order_Date'] = df_order['order_date'].apply(lambda x: Convert_Date(x))
df_order['Source_1'] = df_order['source_1'].apply(lambda x: int(x[11:]))
df_order['Source_2'] = df_order['source_2'].apply(lambda x: int(x[11:]))
df_order['Unit'] = df_order['unit'].apply(lambda x: int(x[11:]))
df_order['PreDays'] = (df_order['Begin_Date'] - df_order['Order_Date']).dt.days
df_order['Begin_Date_Weekday'] = df_order['Begin_Date'].dt.dayofweek
df_order['Order_Date_Weekday'] = df_order['Order_Date'].dt.dayofweek
df_order['Return_Date_Weekday'] = (df_order['Begin_Date'].dt.dayofweek + df_order['days']) % 7
df_order['Order_Date_Year'] = df_order['Order_Date'].dt.year
df_order['Begin_Date_Year'] = df_order['Begin_Date'].dt.year
df_order['Order_Date_Month'] = df_order['Order_Date'].dt.month
df_order['Begin_Date_Month'] = df_order['Begin_Date'].dt.month
df_order['Order_Date_Day'] = df_order['Order_Date'].dt.day
df_order['Begin_Date_Day'] = df_order['Begin_Date'].dt.day
df_order['Order_Date_Week'] = df_order['Order_Date'].dt.week
df_order['Begin_Date_Week'] = df_order['Begin_Date'].dt.week
df_order['Order_Date_Quarter'] = df_order['Order_Date'].dt.quarter
df_order['Begin_Date_Quarter'] = df_order['Begin_Date'].dt.quarter

df_order.to_csv('df_order.csv', index=False)
