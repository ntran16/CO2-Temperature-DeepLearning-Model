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

inputSeries = num2cell(zx1);
targetSeries = num2cell(zt);

for i = 1:10
% Create a Nonlinear Autoregressive Network with External Input
inputDelays = 1:i;
feedbackDelays = 1:4;
hiddenLayerSize = 10;
net = narxnet(inputDelays,feedbackDelays,hiddenLayerSize);

[inputs,inputStates,layerStates,targets] = ... 
    preparets(net,inputSeries,{},targetSeries);

% Set up Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% Initialize weight, choose training function, choose performance score, train the Network, set max train to 200
net = init(net);
net.trainFcn = 'trainbr'; % Levenberg-Marquardt
net.performFcn = 'mse'; % Mean squared error
net.trainParam.epochs = 200;	
[net,tr] = train(net,inputs,targets,inputStates,layerStates);

% Test the Network
outputs = net(inputs,inputStates,layerStates);
errors = gsubtract(targets,outputs);
performance(i) = perform(net,targets,outputs)

% Train a closeloop network
netc = closeloop(net);
netc.name = [net.name ' - Closed Loop'];
[xc,xic,aic,tc] = preparets(netc,inputSeries,{},targetSeries);
yc = netc(xc,xic,aic);
closedLoopPerformance(i) = perform(netc,tc,yc)

nets = removedelay(net);
nets.name = [net.name ' - Predict One Step Ahead'];
% view(nets)
[xs,xis,ais,ts] = preparets(nets,inputSeries,{},targetSeries);
ys = nets(xs,xis,ais);
earlyPredictPerformance(i) = perform(nets,ts,ys)
end