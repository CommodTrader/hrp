import os

import pandas as pd
import feather

ticker_list = [
                   'C',
                   'W',
                   'KW',
                    'S',
                    'SM',
                    'BO',
                    'LC',
                    'LH',
                   'HG',
                   'CL',
                   'CO',
                   'GO',
                    'HO',
                    'XB',
                   'NG',
                   'SB',
                   'CC',
                   'KC'
                   ]


DATA_DIR = 'X:\data\continuous_time_series'

file_suffix = '_flat_price_on_fnd_continuous.feather'

ts_ = []
for ticker in ticker_list:
    file_name = f'{ticker}{file_suffix}'
    file_path = os.path.join(DATA_DIR, file_name)
    df_ = feather.read_dataframe(file_path)
    #df_ = pd.read_feather(file_path)
    ts = (df_.set_index('datetime')['dod_change']/df_.set_index('datetime')['close']).to_frame(ticker)
    ts_.append(ts)

df0 = pd.concat(ts_, axis=1)['2004':]

#print(df0)