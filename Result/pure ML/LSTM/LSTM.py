import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# 1. 載入資料
def load_single_file(file):
    with h5py.File("/content/2024-BCI-Intro_DoA-Monitoring/More-features/raw_data/" + file, 'r') as f:
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

model = create_lstm_model((X.shape[1], 1))

# 訓練模型
strategy = tf.distribute.MirroredStrategy()  # 使用所有可用的GPU
with strategy.scope():
    model = create_lstm_model((X.shape[1], 1))
    model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)

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
