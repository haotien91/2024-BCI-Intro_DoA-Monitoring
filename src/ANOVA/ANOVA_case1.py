import h5py
import numpy as np
import pandas as pd
from scipy.stats import f_oneway

# 確認文件路徑
file_path = '../../resources/raw_data/case1.mat'

# 讀取Matlab v7.3 .mat檔案
with h5py.File(file_path, 'r') as f:
    eeg_data = f['EEG'][:]
    bis_data = f['bis'][:]

# 假設BIS取樣頻率為每秒一次
bis_sampling_rate = 1
bis_interval = 5
bis_segments = np.array_split(bis_data[0], len(bis_data[0]) // (bis_interval * bis_sampling_rate))

# EEG取樣頻率為128點每秒
eeg_sampling_rate = 128
eeg_segments = [eeg_data[0][i*eeg_sampling_rate*bis_interval:(i+1)*eeg_sampling_rate*bis_interval] for i in range(len(bis_segments))]

# 確保BIS和EEG數據片段數量一致
assert len(bis_segments) == len(eeg_segments)

# 打印片段數量
print(f"Number of BIS segments: {len(bis_segments)}")
print(f"Number of EEG segments: {len(eeg_segments)}")

# 對每個片段進行ANOVA分析
anova_results = []
for i, (bis_seg, eeg_seg) in enumerate(zip(bis_segments, eeg_segments)):
    # ANOVA分析
    f_val, p_val = f_oneway(bis_seg, eeg_seg)
    anova_results.append((f_val, p_val))

# 轉換為DataFrame
anova_df = pd.DataFrame(anova_results, columns=['F-Value', 'P-Value'])

# 打印所有ANOVA結果
print(anova_df)

# 將ANOVA結果按F值進行排序
sorted_anova_df = anova_df.sort_values(by='F-Value')
print(sorted_anova_df.head(10))  # 顯示F值最低的10個片段
