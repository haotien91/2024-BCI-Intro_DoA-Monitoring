import os
import numpy as np
import h5py
from antropy import sample_entropy
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from joblib import parallel_backend

# 設置資料夾和文件範圍
directory = '../../../resources/raw_data/'
file_template = 'case{}.mat'
file_range = range(1, 25)

# 定義每段EEG數據的長度（5秒，128 Hz）
segment_length = 5 * 128

# 初始化存儲所有文件數據的列表
all_features = []
all_labels = []

for case_num in file_range:
    file_name = file_template.format(case_num)
    print(f'Processing file: {file_name}')
    file_path = os.path.join(directory, file_name)
    
    if os.path.isfile(file_path):
        with h5py.File(file_path, 'r') as f:
            eeg_data = np.array(f['EEG']).flatten()
            bis_data = np.array(f['bis']).flatten()

        # 只使用样本熵作為特徵
        for i in range(0, len(eeg_data), segment_length):
            segment = eeg_data[i:i + segment_length]
            if len(segment) == segment_length:
                # 样本熵
                sampen = sample_entropy(segment)
                all_features.append([sampen])

        # 將對應的BIS值添加到標籤列表中
        all_labels.extend(bis_data[:len(all_features) - len(all_labels)])

# 將所有特徵和標籤數據轉換為NumPy數組
X = np.array(all_features)
y = np.array(all_labels)

# 標準化特徵數據
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 切分數據集為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 設置隨機森林模型的最佳參數
best_params = {'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 300}
best_rf = RandomForestRegressor(random_state=42, **best_params)

print("Training model with best parameters...")
# 使用最佳參數訓練模型
best_rf.fit(X_train, y_train)

print("Making predictions and evaluating the model...")
# 預測並評估模型
y_pred = best_rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# 計算準確度
tolerance = 2  # 設置容差範圍 = 2
correct_predictions = np.abs(y_pred - y_test) <= tolerance
accuracy = np.mean(correct_predictions)

print(f'Mean Squared Error: {mse}')
print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Best Parameters: {best_params}')

# 顯示預測結果與真實值的比較，並標示準確度
plt.figure(figsize=(10, 5))
plt.plot(y_test, label='True BIS')
plt.plot(y_pred, label='Predicted BIS')
plt.legend()
plt.xlabel('Sample Index')
plt.ylabel('BIS Value')
plt.title(f'True BIS vs Predicted BIS (Accuracy: {accuracy * 100:.2f}%)')
plt.savefig('./Results/Prediction-SampEn.png')
plt.show()
