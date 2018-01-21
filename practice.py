import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
print(os.getcwd())

filepath = 'PRSA_data_2010.1.1-2014.12.31.csv'
data = pd.read_csv(filepath, index_col=0)
print(data.head())


# 日付をdatetime型として１つにまとめて，dataframeのindexとする

from datetime import datetime
#print(datetime(2000,1,12,3,33,33))
data.index = list(map(datetime, data['year'], data['month'], data['day'], data['hour']))
data.index.name = 'date'
data = data.drop(['year', 'month', 'day', 'hour'], axis=1)
data.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
print(data.head())

# 欠損値がある行を削除
data = data.dropna()
print(data.head())


# wnd_dirだけ，数値ではない！
# カテゴリカルな変数を，整数値にエンコーディングする
from sklearn.preprocessing import LabelEncoder
#print(data['wnd_dir'].head())
label_enc = LabelEncoder()
wnd_dir = label_enc.fit_transform(data['wnd_dir'])
data['wnd_dir'] = wnd_dir
print(data.head())

# 整形したデータを保存
data.to_csv('pollution.csv')



###################################################################################################
# データの読み込み
filepath = 'pollution.csv'
data = pd.read_csv(filepath, index_col=0)
print(data.head())
#print(data.ix[:,0])

# データのプロット
# wnd_dirだけ，数値ではない！
import dateutil.parser
timestamps = []
for i in range(len(data)):
    timestamps.append(dateutil.parser.parse(data.index[i]))

num_cols = len(data.columns)
plt.figure()
for i in range(num_cols):
    plt.subplot(num_cols, 1, i+1)
    plt.plot(timestamps, data.ix[:,i])
    plt.title(data.columns[i], y=0.5, loc='right')
plt.show()

print(data.shift(1).head())

print([i for i in range(1,2)])
print(data.columns[:])

# 時系列予測するために，時系列テーブルを作成
def dataframe2timetable(dataframe, obj_col=0, delay_length=1):
    colnames_org = dataframe.columns
    timetable = pd.DataFrame(dataframe.ix[:,obj_col])
    for i in range(1, delay_length+1):
        colnames = ['%s(t-%d)'%(col, i) for col in colnames_org]
        d_shift = dataframe.shift(i)
        d_shift.columns = colnames

        timetable = pd.concat([timetable, d_shift], axis=1)

    timetable = timetable.dropna()

    return timetable

timetable = dataframe2timetable(data, delay_length=2)
print(timetable.head())


# 各データを標準化
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
data_scaled = scalar.fit_transform(data)
print(data_scaled[:10])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_scaled[:,1:], data_scaled[:,0], test_size=0.2, random_state=0)
print(X_train.shape)

# design model
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.recurrent import LSTM

length_of_sequence = X_train.shape[1]
look_back = 2
n_hidden = 50

model = Sequential()
model.add(LSTM(n_hidden, input_shape=(length_of_sequence, look_back)))
# batch_input_shape:  LSTMに入力するデータの形を指定([バッチサイズ，step数，特徴の次元数]を指定
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

# fit network
history = model.fit(X_train, y_train, epochs=50, batch_size=72, validation_split=0.1)
