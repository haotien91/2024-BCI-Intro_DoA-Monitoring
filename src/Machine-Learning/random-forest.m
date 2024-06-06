% 設置參數
fs = 128; % 採樣頻率
n_points_per_bis = 5 * fs;

% 初始化變數以儲存所有訓練數據
X_train = [];
Y_train = [];

% 載入case1.mat 到 case18.mat 檔案
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
    mean_fft = mean(eeg_segments);
    std_fft = std(eeg_segments);
    max_fft = max(eeg_segments);
    min_fft = min(eeg_segments);
    
    % 構建特徵集
    features = [mean_fft; std_fft; max_fft; min_fft]';
    
    % 構建訓練集
    X_train = [X_train; features];
    Y_train = [Y_train; bis(1:num_segments)];
end

% 使用網格搜索調整隨機森林模型的參數
numTreesOptions = [50, 100, 200];
minLeafSizeOptions = [5, 10, 20, 30];
bestAccuracy = inf;
bestNumTrees = 70;
bestMinLeafSize = 5;

for numTrees = numTreesOptions
    for minLeafSize = minLeafSizeOptions
        % 構建Random Forest模型
        Mdl = TreeBagger(numTrees, X_train, Y_train, 'Method', 'regression', 'MinLeafSize', minLeafSize);
        
        % 初始化變數以儲存所有測試數據
        X_test = [];
        Y_test = [];
        Y_pred = [];
        
        % 載入case19.mat 到 case21.mat 檔案
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
            mean_fft = mean(eeg_segments);
            std_fft = std(eeg_segments);
            max_fft = max(eeg_segments);
            min_fft = min(eeg_segments);
            
            % 構建特徵集
            features = [mean_fft; std_fft; max_fft; min_fft]';
            
            % 構建測試集
            X_test = [X_test; features];
            Y_test = [Y_test; bis(1:num_segments)];
            
            % 使用模型預測BIS值
            Y_pred = [Y_pred; predict(Mdl, features)];
        end
        
        % 計算準確度
        accuracy = mean(abs(Y_pred - Y_test) ./ Y_test);
        
        % 更新最佳參數
        if accuracy < bestAccuracy
            bestAccuracy = accuracy;
            bestNumTrees = numTrees;
            bestMinLeafSize = minLeafSize;
        end
    end
end

% 使用最佳參數重新訓練模型
Mdl = TreeBagger(bestNumTrees, X_train, Y_train, 'Method', 'regression', 'MinLeafSize', bestMinLeafSize);

% 初始化變數以儲存所有測試數據
X_test = [];
Y_test = [];
Y_pred = [];

% 載入case22.mat 到 case24.mat 檔案
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
    mean_fft = mean(eeg_segments);
    std_fft = std(eeg_segments);
    max_fft = max(eeg_segments);
    min_fft = min(eeg_segments);
    
    % 構建特徵集
    features = [mean_fft; std_fft; max_fft; min_fft]';
    
    % 構建測試集
    X_test = [X_test; features];
    Y_test = [Y_test; bis(1:num_segments)];
    
    % 使用模型預測BIS值
    Y_pred = [Y_pred; predict(Mdl, features)];
end

% 計算準確度
MAPE = mean(abs((Y_pred - Y_test) ./ Y_test));

% 繪製預測值與真實值
figure;
plot(Y_test, 'b');
hold on;
plot(Y_pred, 'r');
legend('True BIS', 'Predicted BIS');
xlabel('Time (5s intervals)');
ylabel('BIS');
title(['BIS Prediction with Random Forest (MAPE: ' num2str(MAPE*100) '%)']);
hold off;
