import numpy as np
import h5py
import pandas as pd
from scipy.signal import welch
from antropy import sample_entropy
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 讀取.mat檔案
with h5py.File('./raw_data/case1.mat', 'r') as f:
    eeg_data = np.array(f['EEG']).flatten()
    bis_data = np.array(f['bis']).flatten()

# 定義每段EEG數據的長度（5秒，128 Hz）
segment_length = 5 * 128

# 計算每段EEG數據的特徵：樣本熵、統計特徵、頻域特徵
features = []
for i in range(0, len(eeg_data), segment_length):
    segment = eeg_data[i:i + segment_length]
    if len(segment) == segment_length:
        # 樣本熵
        sampen = sample_entropy(segment)
        
        # 統計特徵
        mean_val = np.mean(segment)
        std_val = np.std(segment)
        
        # 頻域特徵（使用Welch方法計算功率譜密度）
        freqs, psd = welch(segment, fs=128)
        psd_mean = np.mean(psd)
        psd_std = np.std(psd)
        
        features.append([sampen, mean_val, std_val, psd_mean, psd_std])

# 構建特徵和標籤數據集
X = np.array(features)
y = bis_data[:len(features)]  # 對應的BIS值

# 標準化特徵數據
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 切分數據集為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 設置隨機森林模型的超參數網格
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 使用網格搜索和交叉驗證選擇最佳模型參數
rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# 使用最佳參數訓練模型
best_rf = grid_search.best_estimator_
best_rf.fit(X_train, y_train)

# 預測並評估模型
y_pred = best_rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# 計算準確度
tolerance = 5  # 設置容差範圍
correct_predictions = np.abs(y_pred - y_test) <= tolerance
accuracy = np.mean(correct_predictions)

print(f'Mean Squared Error: {mse}')
print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Best Parameters: {grid_search.best_params_}')

# 顯示預測結果與真實值的比較
plt.figure(figsize=(10, 5))
plt.plot(y_test, label='True BIS')
plt.plot(y_pred, label='Predicted BIS')
plt.legend()
plt.xlabel('Sample Index')
plt.ylabel('BIS Value')
plt.title('True BIS vs Predicted BIS')
plt.show()

# 繪製0/1圖
correct_predictions_binary = correct_predictions.astype(int)

plt.figure(figsize=(10, 5))
plt.plot(correct_predictions_binary, label='Correct Predictions (1) / Incorrect Predictions (0)')
plt.ylim(-0.1, 1.1)
plt.xlabel('Sample Index')
plt.ylabel('Prediction Accuracy')
plt.title('Prediction Accuracy (1 for correct, 0 for incorrect)')
plt.legend()
plt.show()
