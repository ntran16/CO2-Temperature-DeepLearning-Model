clear all;
close all;
clc;

A = csvread('D:\Dropbox\GPL\training.csv', 1);
% co2 = A(:,1);
% nw701 = A(:,2);
% co2 = co2';
% nw701 = nw701';

co2 = A(:,1);
nw701 = A(:,2);
t = co2';
x = nw701';
zx1 = zscore(x,1);
zt = zscore(t,1);

inputSeries = num2cell(zx1,1);
targetSeries = num2cell(zt,1);

for i = 1:2:100
    for j = 10
rng('default')
% Create a Nonlinear Autoregressive Network with External Input
feedbackDelays = 1:i;
hiddenLayerSize = j;

net = narnet(feedbackDelays,hiddenLayerSize);
[Xs,Xi,Ai,targets] = preparets(net,{},{},targetSeries);

% Set up Division of Data for Training, Validation, Testing
net.divideFcn = 'divideblock';
net.divideMode = 'time';  % Divide up every value
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% Initialize weight, choose training function, choose performance score, train the Network, set max train to 200
net = init(net);
net.trainFcn = 'trainlm'; % Levenberg-Marquardt
net.performFcn = 'mse'; % Mean squared error
net.trainParam.epochs = 200;	
[net tr] = train(net,Xs,targets,Xi,Ai);

% Test the Network
outputs = net(Xs,Xi);
errors = gsubtract(targets,outputs);
perf(i,j) = perform(net,targets,outputs);

% Training score
trainTargets = gmultiply(targets,tr.trainMask);
trainTargets = cell2mat(trainTargets);
trainTargets=[trainTargets(~isnan(trainTargets))];
trainPerformance(i,j) = tr.best_perf;
train_mse(i,j) = mean(var(trainTargets,1));

%Testing score
testTargets = gmultiply(targets,tr.testMask);
testTargets = cell2mat(testTargets);
testTargets=[trainTargets(~isnan(trainTargets))];
testPerformance(i,j) = tr.best_tperf;
test_mse(i,j) = mean(var(testTargets,1));

%Validation score
valTargets = gmultiply(targets,tr.valMask);
valTargets = cell2mat(valTargets);
valTargets=[trainTargets(~isnan(trainTargets))];
valPerformance(i,j) = tr.best_vperf;
val_mse(i,j) = mean(var(valTargets,1));
    end
end