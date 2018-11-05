import numpy as np
import pandas as pd

df_test = pd.read_csv('cmp4_sample_submission.csv')
df_test['deal_or_not'] = np.random.rand(len(df_test))

df_test.to_csv('random_submission.csv', index=False)