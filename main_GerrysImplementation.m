%%  Gerry Chen
%   digit/train.m  -  training script for classifying handwritten digits

%% Setup
% clear;clc;
addpath('./CNN/')
load('MNIST/trainingData');
numSamples = length(labels);
categories = sort(unique(labels));
indsByCat = {};
for cat = categories
    indsByCat{end+1} = find(labels==cat);
end

net = CNN(size(images(:,:,1)), length(categories));

%% train
figure(2);clf;colormap gray;
figure(3);clf;colormap gray
allRes = [];
    
% figure(4);clf;colormap gray
% for i = 1:10
%     subplot(2,5,i);
%     image(images(:,:,set(i)));
% end
    
for iters = 1:300

    mult = 1;
    set = zeros(mult*length(categories),1);
    outs = zeros(numel(set),length(categories));
    residuals = zeros(size(outs));
    for i = 1:length(categories)
        set(mult*(i-1)+1 : mult*i) = datasample(indsByCat{i}, mult);
    end

    for i = 1:length(set)
        outs(i,:) = net.evaluate(images(:,:,set(i)),true);
        residuals(i,:) = 2*(labels(set(i))==categories)-1 - outs(i,:);
    end
%     dispRes(labels(set), residuals)
    fprintf('meanRes: %.2f\n', norm(residuals));
    allRes = [allRes, norm(residuals)];
    net.backPropogate(residuals, [], .0003);
    if (mod(iters,30) == 0)
        figure(2);
        imshow(residuals',[-CNN.A, CNN.A], 'InitialMagnification','fit',...
            'XData',[0,size(residuals,1)],'YData',[0,size(residuals,2)]);
        axis square
        figure(3);
        for i = 1:10
            subplot(2,5,i)
            imagesc(reshape(net.Layers{1}.weights(i,:),[28,28]));
        end
        drawnow()
    end
end

%% test/visually observe
figure(1);clf;
plot(allRes)

testSet = zeros(length(categories),1);
testOuts = zeros(length(categories));
testResiduals = zeros(length(categories));
for i = 1:length(categories)
    testSet(i) = datasample(indsByCat{i}, 1);
    testOuts(i,:) = net.evaluate(images(:,:,testSet(i)), false);
    testResiduals(i,:) = 2*(labels(testSet(i))==categories)-1 - testOuts(i,:);
end
dispRes(labels(testSet), testOuts)
dispRes(labels(testSet), testResiduals)

%% test success rate
% [SSR,incorrect] = testNN(net);
% fprintf('success rate: %.1f%%\n', SSR*100);


%% debug functions
function [] = dispRes(answers, residuals)
    fprintf('  \t');
    fprintf('  %d   ',1:size(residuals,2));
    fprintf('\n');
    for i = 1:length(answers)
        fprintf('%d:\t', answers(i));
        fprintf('%+.2f ', residuals(i,:));
        fprintf('\n');
    end
end