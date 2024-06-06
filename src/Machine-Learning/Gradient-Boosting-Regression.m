% 設置參數
fs = 128; % 採樣頻率
n_points_per_bis = 5 * fs;

% 初始化變數以儲存所有訓練數據
X_train = [];
Y_train = [];

% 初始化變數以儲存所有驗證數據
X_val = [];
Y_val = [];

% 載入case1.mat 到 case18.mat 檔案作為訓練數據
for i = 1:18
    filename = sprintf('../../resources/raw_data/case%d.mat', i);
    data = load(filename);
    
    % 獲取EEG和BIS變數
    if isfield(data, 'EEG') && isfield(data, 'bis')
        EEG = data.EEG;
        bis = data.bis;
    else
        error('檔案%s中未找到變數EEG或bis', filename);
    end
    
    % 進行傅立葉轉換
    eeg_fft = abs(fft(EEG));
    
    % 匹配每個BIS點與其對應的頻譜
    num_segments = floor(length(EEG) / n_points_per_bis);
    eeg_segments = reshape(eeg_fft(1:num_segments*n_points_per_bis), n_points_per_bis, num_segments);
    
    % 提取頻譜的統計特徵
    mean_fft = mean(eeg_segments, 1);
    std_fft = std(eeg_segments, 0, 1);
    max_fft = max(eeg_segments, [], 1);
    min_fft = min(eeg_segments, [], 1);
    
    % 構建特徵集
    features = [mean_fft; std_fft; max_fft; min_fft]';
    
    % 將數據加入訓練集
    X_train = [X_train; features];
    Y_train = [Y_train; bis(1:num_segments)];
end

% 載入case19.mat 到 case21.mat 檔案作為驗證數據
for i = 19:21
    filename = sprintf('../../resources/raw_data/case%d.mat', i);
    data = load(filename);
    
    % 獲取EEG和BIS變數
    if isfield(data, 'EEG') && isfield(data, 'bis')
        EEG = data.EEG;
        bis = data.bis;
    else
        error('檔案%s中未找到變數EEG或bis', filename);
    end
    
    % 進行傅立葉轉換
    eeg_fft = abs(fft(EEG));
    
    % 匹配每個BIS點與其對應的頻譜
    num_segments = floor(length(EEG) / n_points_per_bis);
    eeg_segments = reshape(eeg_fft(1:num_segments*n_points_per_bis), n_points_per_bis, num_segments);
    
    % 提取頻譜的統計特徵
    mean_fft = mean(eeg_segments, 1);
    std_fft = std(eeg_segments, 0, 1);
    max_fft = max(eeg_segments, [], 1);
    min_fft = min(eeg_segments, [], 1);
    
    % 構建特徵集
    features = [mean_fft; std_fft; max_fft; min_fft]';
    
    % 將數據加入驗證集
    X_val = [X_val; features];
    Y_val = [Y_val; bis(1:num_segments)];
end

% 進行交叉驗證來微調模型參數
cv = cvpartition(size(X_train, 1), 'KFold', 5);
min_loss = inf;
best_num_learners = 0;

for num_learners = 50:50:500
    model = fitrensemble(X_train, Y_train, 'Method', 'LSBoost', 'NumLearningCycles', num_learners, 'CrossVal', 'on', 'CVPartition', cv);
    loss = kfoldLoss(model);
    if loss < min_loss
        min_loss = loss;
        best_num_learners = num_learners;
    end
end

% 使用最佳參數重新訓練模型
gbrModel = fitrensemble(X_train, Y_train, 'Method', 'LSBoost', 'NumLearningCycles', best_num_learners);

% 初始化變數以儲存所有測試數據
X_test = [];
Y_test = [];
Y_pred = [];

% 載入case22.mat 到 case24.mat 檔案作為測試數據
for i = 22:24
    filename = sprintf('../../resources/raw_data/case%d.mat', i);
    data = load(filename);
    
    % 獲取EEG和BIS變數
    if isfield(data, 'EEG') && isfield(data, 'bis')
        EEG = data.EEG;
        bis = data.bis;
    else
        error('檔案%s中未找到變數EEG或bis', filename);
    end
    
    % 進行傅立葉轉換
    eeg_fft = abs(fft(EEG));
    
    % 匹配每個BIS點與其對應的頻譜
    num_segments = floor(length(EEG) / n_points_per_bis);
    eeg_segments = reshape(eeg_fft(1:num_segments*n_points_per_bis), n_points_per_bis, num_segments);
    
    % 提取頻譜的統計特徵
    mean_fft = mean(eeg_segments, 1);
    std_fft = std(eeg_segments, 0, 1);
    max_fft = max(eeg_segments, [], 1);
    min_fft = min(eeg_segments, [], 1);
    
    % 構建特徵集
    features = [mean_fft; std_fft; max_fft; min_fft]';
    
    % 構建測試集
    X_test = [X_test; features];
    Y_test = [Y_test; bis(1:num_segments)];
    
    % 使用模型預測BIS值
    Y_pred = [Y_pred; predict(gbrModel, features)];
end

% 計算準確度
MAPE = mean(abs((Y_pred - Y_test) ./ Y_test)) * 100;

% 繪製預測值與真實值
figure;
plot(Y_test, 'b');
hold on;
plot(Y_pred, 'r');
legend('True BIS', 'Predicted BIS');
xlabel('Time (5s intervals)');
ylabel('BIS');
title(['BIS Prediction with Gradient Boosting (MAPE: ' num2str(MAPE) '%)']);
hold off;