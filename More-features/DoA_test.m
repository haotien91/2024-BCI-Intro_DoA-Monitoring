duration = 3800;
sample_rate = 128;

% EEG in 50s
% 
points = sample_rate * duration

dataSegment = EEG(1:points);
timeVector = linspace(0, duration, points);

figure;
plot(timeVector, dataSegment);
xlabel('Time (seconds)');
ylabel('EEG Amplitude');
title('EEG Data for 3800 Seconds (whole trail)');


% BIS in 50s
bis_period = 5;

% 計算總點數
points = sample_rate * duration;

% 初始化 BIS 數據向量
dataSegment = zeros(1, points);

% 計算每個周期的點數
points_per_period = sample_rate * bis_period;

% 為每個周期分配對應的 BIS 值
for k = 0:ceil(duration / bis_period) - 1
    index_start = k * points_per_period + 1;
    index_end = min((k + 1) * points_per_period, points);
    dataSegment(index_start:index_end) = bis(k + 1);  % 假設 bis 是一個已定義的函數或數組
end

% 創建時間向量
timeVector = linspace(0, duration, points);

% 繪圖
figure;
plot(timeVector, dataSegment);  % 繪製數據
xlabel('Time (seconds)');  % x軸標籤
ylabel('BIS value');  % y軸標籤
title('BIS value for 3800 Seconds (whole trial)');  % 圖表標題
xlim([0 3800]);  % x軸範圍
ylim([0 100]);  % y軸範圍
yticks(0:10:100);  % 設定y軸的刻度

% 使用fill函數填滿40到60的區域
hold on;
xLimits = xlim;
fill([xLimits(1) xLimits(2) xLimits(2) xLimits(1)], [40 40 60 60], 'm', 'FaceAlpha', 0.2);
% 在填滿區域正中心寫下 "最佳麻醉深度"
text(mean(xLimits), 50, 'Optimal DoA', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 10);
% 在y=0和y=100的確切位置添加紅色文字
text(xLimits(2), 0, 'No EEG consiousness', 'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom', 'FontSize', 10, 'Color', 'red');
text(xLimits(2), 100, 'Fully awake', 'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', 'FontSize', 10, 'Color', 'red');
hold off;
