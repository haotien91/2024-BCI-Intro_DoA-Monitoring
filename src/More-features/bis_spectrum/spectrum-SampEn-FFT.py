import os
import numpy as np
import h5py
from antropy import sample_entropy
from scipy.signal import welch
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from joblib import parallel_backend

# 列出目錄中的文件
directory = '../../../resources/spectrum_data/'
files = os.listdir(directory)
print("Files in directory:", files)

# 定義一個函數來讀取和處理每個檔案
def process_file(case_num):
    file_name = os.path.join(directory, f'spectrum_case{case_num}.mat')
    if not os.path.isfile(file_name):
        print(f"File not found: {file_name}")
        return None, None
    
    try:
        with h5py.File(file_name, 'r') as f:
            bis_data = np.array(f['processed_bis']).flatten()
            eeg_data = np.array(f['processed_EEG']).flatten()
        return bis_data, eeg_data
    except Exception as e:
        print(f"Error reading file {file_name}: {e}")
        return None, None

print("Start processing files...")
# 將所有檔案的數據合併
all_bis_data = []
all_eeg_data = []
for case_num in range(1, 25):
    bis_data, eeg_data = process_file(case_num)
    if bis_data is not None and eeg_data is not None:
        all_bis_data.append(bis_data)
        all_eeg_data.append(eeg_data)

if not all_bis_data or not all_eeg_data:
    raise ValueError("No valid BIS or EEG data found.")

print("Files processed. Combining data...")
# 合併所有的 BIS 和 EEG 數據
all_bis_data = np.concatenate(all_bis_data)
all_eeg_data = np.concatenate(all_eeg_data)

# 定義每段EEG數據的長度（5秒，128 Hz）
segment_length = 5 * 128

# 計算每段EEG數據的特徵：樣本熵和頻域特徵
print("Calculating features from EEG data...")
features = []
for i in range(0, len(all_eeg_data), segment_length):
    segment = all_eeg_data[i:i + segment_length]
    if len(segment) == segment_length:
        # 樣本熵
        sampen = sample_entropy(segment)
        
        # 頻域特徵（使用Welch方法計算功率譜密度）
        freqs, psd = welch(segment, fs=128)
        psd_mean = np.mean(psd)
        psd_std = np.std(psd)
        
        features.append([sampen, psd_mean, psd_std])

print("Features calculated. Constructing feature and label datasets...")
# 構建特徵和標籤數據集
X = np.array(features)
y = all_bis_data[:len(features)]  # 對應的BIS值

print("Standardizing features...")
# 標準化特徵數據
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Applying PCA for dimensionality reduction...")
# 使用 PCA 降維
n_components = min(len(X_scaled), X_scaled.shape[1]) - 1  # 確保 n_components 不超過允許範圍
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

print("Splitting dataset into training and test sets...")
# 切分數據集為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# 設置隨機森林模型的超參數網格
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

print("Starting grid search for hyperparameter tuning...")
# 使用網格搜索和交叉驗證選擇最佳模型參數
with parallel_backend('threading', n_jobs=40):
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

print("Training model with best parameters...")
# 使用最佳參數訓練模型
best_rf = grid_search.best_estimator_
best_rf.fit(X_train, y_train)

print("Making predictions and evaluating the model...")
# 預測並評估模型
y_pred = best_rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# 計算準確度
tolerance = 0.5  # 設置容差範圍
correct_predictions = np.abs(y_pred - y_test) <= tolerance
accuracy = np.mean(correct_predictions)

print(f'Mean Squared Error: {mse}')
print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Best Parameters: {grid_search.best_params_}')

print("Plotting results...")
# 顯示預測結果與真實值的比較，並標示準確度
plt.figure(figsize=(10, 5))
plt.plot(y_test, label='True BIS')
plt.plot(y_pred, label='Predicted BIS')
plt.legend()
plt.xlabel('Sample Index')
plt.ylabel('BIS Value')
plt.title(f'True BIS vs Predicted BIS (Accuracy: {accuracy * 100:.2f}%)')
plt.savefig('./Results/Prediction-SampEn-FFT.png')
plt.show()

# 計算混淆矩陣
y_pred_rounded = np.round(y_pred).astype(int)
y_test_rounded = np.round(y_test).astype(int)

conf_matrix = confusion_matrix(y_test_rounded, y_pred_rounded, labels=[-4, -3, -2, -1, 0, 1, 2, 3, 4])
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[-4, -3, -2, -1, 0, 1, 2, 3, 4])

# 繪製並保存混淆矩陣
plt.figure(figsize=(10, 10))
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.savefig('./Results/Figures/confusion_matrix-SampEn-FFT.png')
plt.show()
