clear all;
close all;
clc;

A = csvread('D:\Dropbox\GPL\training.csv', 1);
% co2 = A(:,1);
% nw701 = A(:,2);
% co2 = co2';
% nw701 = nw701';
zx1 = []
for i = (2:8)
co2 = A(:,1);
nw701 = A(:,i);
t = co2';
x = nw701';
zzx1 = zscore(x,1);
zt = zscore(t,1);

[acor,lag] = xcorr(zzx1,zt, 'coeff');
[~,I(i)] = max(abs(acor));
lagDiff(i) = lag(I(i));

zx1 = [zx1;zzx1];
% 
% figure (i-1)
% plot(lag,acor)
end

[acor,lag] = xcorr(zt,zt, 'coeff');
[~,I(1)] = max(abs(acor));
lagDiff(1) = lag(I(1));
plot(lag,acor)

% [acor,lag] = xcorr(zx1,zt, 'coeff');
% 
% figure(2)
% plot(lag,acor)

% a = finddelay(zx1, zt)

% t1 = (0:length(co2)-1);
% t2 = (0:length(nw701)-1);
% figure (2);
% subplot(2,1,1);
% plot(t1,co2);
% title('s_1');
% 
% subplot(2,1,2);
% plot(t2,nw701);
% title('s_2');
% xlabel('Time (s)')

% target = co2(1:5000,:)
% input = nw701(1:5000,:)
% input = zscore(input,1);
% target = zscore(target,1);
% 
% [acor,lag] = xcorr(input,target, 'coeff');
% [~,I] = max(abs(acor));
% lagDiff = lag(I);
% % 
% figure(1)
% plot(lag,acor)
% 
% t1 = (0:length(input)-1);
% t2 = (0:length(target)-1);
% figure (2);
% subplot(2,1,1);
% plot(t1,input);
% title('nw701');
% 
% subplot(2,1,2);
% plot(t2,target);
% title('target');
% xlabel('Time (s)')
