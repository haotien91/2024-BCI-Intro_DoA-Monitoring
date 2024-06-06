% 批量處理文件
for case_num = 1:24
    % 生成文件名
    file_name = sprintf('./raw_data/case%d.mat', case_num);
    
    % 讀取.mat檔案
    data = load(file_name);
    bis_data = data.bis;
    eeg_data = data.EEG;
    
    % 將EEG數據剪到5秒的倍數
    segment_length = 5 * 128;
    eeg_data = eeg_data(1:floor(length(eeg_data) / segment_length) * segment_length);
    
    % 計算EEG數據的總秒數
    eeg_seconds = length(eeg_data) / 128;
    
    % 計算BIS數據的總秒數
    bis_seconds = length(bis_data) * 5;
    
    % 比較EEG秒數和BIS秒數，取較小的
    total_seconds = min(eeg_seconds, bis_seconds);
    
    % 計算總共的EEG資料點數和BIS數據點數
    total_eeg_points = total_seconds * 128;
    total_bis_points = total_seconds / 5;
    
    % 剪除多餘的EEG和BIS數據
    eeg_data = eeg_data(1:total_eeg_points);
    bis_data = bis_data(1:total_bis_points);
    
    % 定義處理後的BIS值
    processed_bis = zeros(size(bis_data));
    
    % 根據規則進行數據前處理
    for i = 1:length(bis_data)
        if bis_data(i) >= 40 && bis_data(i) < 60
            processed_bis(i) = 0;
        elseif bis_data(i) >= 30 && bis_data(i) < 40
            processed_bis(i) = -1;
        elseif bis_data(i) >= 20 && bis_data(i) < 30
            processed_bis(i) = -2;
        elseif bis_data(i) >= 10 && bis_data(i) < 20
            processed_bis(i) = -3;
        elseif bis_data(i) >= 0 && bis_data(i) < 10
            processed_bis(i) = -4;
        elseif bis_data(i) >= 60 && bis_data(i) < 70
            processed_bis(i) = 1;
        elseif bis_data(i) >= 70 && bis_data(i) < 80
            processed_bis(i) = 2;
        elseif bis_data(i) >= 80 && bis_data(i) < 90
            processed_bis(i) = 3;
        elseif bis_data(i) >= 90 && bis_data(i) < 100
            processed_bis(i) = 4;
        end
    end
    
    % 計算EEG數據的偏差
    processed_EEG = zeros(size(eeg_data));
    for i = 1:total_bis_points
        segment_start = (i - 1) * segment_length + 1;
        segment_end = i * segment_length;
        segment = eeg_data(segment_start:segment_end);
        segment_mean = mean(segment);
        processed_EEG(segment_start:segment_end) = segment - segment_mean;
    end
    
    % 生成保存文件名
    save_file_name = sprintf('./spectrum_data/spectrum_case%d.mat', case_num);
    
    % 保存處理後的BIS數據和對應的EEG數據
    save(save_file_name, 'processed_bis', 'processed_EEG', '-v7.3'); % 指定使用HDF5格式保存
end