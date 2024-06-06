import os
import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

# 1. 載入資料
def load_single_file(file):
    with h5py.File("../../resources/raw_data/" + file, 'r') as f:
        eeg = np.array(f['EEG']).flatten()
        bis = np.array(f['bis']).flatten()
    return eeg, bis

def load_data(files, n_jobs=4):
    results = Parallel(n_jobs=n_jobs)(delayed(load_single_file)(file) for file in files)
    eeg_data, bis_data = zip(*results)
    return np.concatenate(eeg_data), np.concatenate(bis_data)

files = ['case1.mat', 'case2.mat', 'case3.mat', 'case4.mat', 'case5.mat',
         'case6.mat', 'case8.mat', 'case9.mat', 'case10.mat', 'case11.mat',
         'case12.mat', 'case13.mat', 'case14.mat', 'case15.mat', 'case16.mat',
         'case17.mat', 'case18.mat', 'case19.mat', 'case20.mat', 'case21.mat']

eeg, bis = load_data(files)

# 2. 傅立葉轉換
def perform_fft(eeg, sampling_rate=128):
    fft_data = np.abs(np.fft.fft(eeg))
    freqs = np.fft.fftfreq(len(eeg), 1 / sampling_rate)
    return fft_data, freqs

fft_data, freqs = perform_fft(eeg)

# 3. LSTM建模
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# 準備LSTM輸入資料
def prepare_data(fft_data, bis, time_step=128 * 5):
    X, y = [], []
    for i in range(0, len(fft_data) - time_step, time_step):
        X.append(fft_data[i:i + time_step])
        y.append(bis[i // time_step])
    return np.array(X), np.array(y)

X, y = prepare_data(fft_data, bis)

# Reshape data for LSTM [samples, time_steps, features]
X = X.reshape(X.shape[0], X.shape[1], 1)

# 使用tf.data API创建数据集
batch_size = 32
train_size = int(0.8 * len(X))
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(buffer_size=1024).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# 创建并训练模型
with tf.device('/GPU:0'):
    model = create_lstm_model((X.shape[1], 1))
    model.fit(train_dataset, epochs=50, validation_data=val_dataset)

# 4. 匯入新的資料
new_files = ['case22.mat', 'case23.mat', 'case24.mat']
new_eeg, new_bis = load_data(new_files)

# 5. 對新的eeg資料做傅立葉轉換
new_fft_data, new_freqs = perform_fft(new_eeg)

# 6. 準備新的輸入資料並預測
new_X, new_y = prepare_data(new_fft_data, new_bis)
new_X = new_X.reshape(new_X.shape[0], new_X.shape[1], 1)
predictions = model.predict(new_X)

# 7. 做圖及計算MAPE
plt.figure(figsize=(12, 6))
plt.plot(predictions, label='Predicted BIS')
plt.plot(new_y, label='True BIS')
plt.legend()
plt.show()

mape = mean_absolute_percentage_error(new_y, predictions)
print(f'MAPE: {mape}')
