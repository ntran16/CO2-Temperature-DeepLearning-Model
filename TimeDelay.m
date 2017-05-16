clear all;
close all;
clc;


for q = 1
    tic
A = csvread('D:\Dropbox\GPL\training.csv', 1);
% co2 = A(:,1);
% nw701 = A(:,2);
% co2 = co2';
% nw701 = nw701';
region = ['NW101';'NW201';'NW301';'NW401';'NW501';'NW601';'NW701';'NW801'];
co2 = A(:,1);
nw701 = A(:,q+1);
t = co2';
x = nw701';
zx1 = zscore(x,1);
zt = zscore(t,1);

inputSeries = num2cell(zx1,1);
targetSeries = num2cell(zt,1);
rng(0);
bracket = [20];
for p = 1:length(bracket)
for i = 3
    for j = 6
% Create a Nonlinear Autoregressive Network with External Input
inputDelay = (i-1)*40:(i-1)*40+bracket(p);
hiddenLayerSize = j;
state = rng;
net = timedelaynet(inputDelay,hiddenLayerSize);
net = init(net);
net = setwb(net,ones(19,1));
[Xs,Xi,Ai,targets] = preparets(net,inputSeries,targetSeries);

% Set up Division of Data for Training, Validation, Testing
net.divideFcn = 'divideblock';
% net.divideMode = 'time';  % Divide up every value
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 1/100;
net.divideParam.testRatio = 29/100;

% Initialize weight, choose training function, choose performance score, train the Network, set max train to 200
net.trainFcn = 'trainbr'; % Levenberg-Marquardt
net.performFcn = 'mse'; % Mean squared error
% net.trainParam.lr = .001;	
% net.performParam.regularization = 0.5;
% net.trainParam.mu_dec = 0.8;	
% net.trainParam.mu_inc = 1.5;	
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

figure()
plot(0:40:(size(trainPerformance)-1)*40,trainPerformance/mean(val_mse))
xlabel(['Delay (days)'])
ylabel('NMSE')
title(['Validation Error - ', region(q,:), ', ', num2str(bracket(p)+1), ' days range'])

end
toc
end
