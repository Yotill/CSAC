% 创建神经网络
net = feedforwardnet([24 12], 'trainlm');
% net = feedforwardnet([40], 'trainscg');
% net.trainParam.miniBatchSize = 1000000;
% 设置网络参数
net.layers{1}.transferFcn = 'tansig'; % 第一层使用双曲正切S形函数
net.layers{2}.transferFcn = 'tansig'; 
net.layers{3}.transferFcn = 'purelin'; % 第一层使用双曲正切S形函数
% net.layers{3}.transferFcn = 'purelin'; % 输出层使用线性激活函数

% 正则化和性能函数
net.performParam.regularization = 0.01; % 正则化项的系数 L2
net.performFcn = 'msereg'; % 使用均方误差作为性能函数

% 训练参数
net.trainParam.epochs = 1000; % 最大迭代次数
net.trainParam.goal = 0.01; % 训练目标误差
net.trainParam.lr = 1e-4; % 学习率
net.trainParam.mu = 1e-2; % Levenberg-Marquardt算法参数
net.trainParam.min_grad = 1e-2; % 最小梯度
net.trainParam.showWindow = true; % 显示训练窗口
net.trainParam.showCommandLine = true; % 显示命令行输出
% 训练网络
net = train(net, input_Pn', output_Pn');