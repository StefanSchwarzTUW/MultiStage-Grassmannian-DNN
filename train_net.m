%% Neural Network Training for "Recursive CSI Quantization of Time-Correlated MIMO Channels by Deep Learning Classification", S. Schwarz, IEEE SPL 2020
% (c) Stefan Schwarz @ Institute of Telecommunications, TU Wien 2020
% Notice: run NN_training for the corresponding number of TX and RX
% antennas first to generate the required training set
% This file requires the Deep Learning Toolbox

% clc;
% clear all;
% close all;
% 
% nTX = 4; % number of transmit antennas (n)
% nRX = 2;  % number of receive antennas (m)
% CB_size = 2^6; % quantization codebook size
NN = 1e6; % number of training samples
NN_fac = 8/6; % 1/NN_fac is the fraction of samples used for validation
file_name = [num2str(nTX) '_' num2str(nRX) '_' num2str(log2(CB_size)) '_' num2str(log10(NN)) '.mat'];  % filename of training-data
load(file_name) % load training data

%% NN structure
drop_prob = 0.001; % drop-out probability of the first layer
net_width = 25;  % width-factor for the first layer
input_layer = imageInputLayer([nTX,nRX,2],'Normalization','none'); 
% input_layer = imageInputLayer([nTX*nRX*2,1],'Normalization','none'); 
layer1 = fullyConnectedLayer(nTX*nRX*2*net_width);
% layer2 = fullyConnectedLayer(nTX*nRX*2*net_width);
% layer3 = fullyConnectedLayer(nTX*nRX*2*net_width/2);
layer4 = fullyConnectedLayer(CB_size); 
layer5 = softmaxLayer;
output_layer = classificationLayer;
% layers = [input_layer,layer1,dropoutLayer(drop_prob),reluLayer,layer2,dropoutLayer(drop_prob),reluLayer,layer3,dropoutLayer(drop_prob),reluLayer,layer4,layer5,output_layer];
% layers = [input_layer,layer1,dropoutLayer(drop_prob),reluLayer,layer2,dropoutLayer(drop_prob),reluLayer,layer4,layer5,output_layer];
layers = [input_layer,layer1,dropoutLayer(drop_prob),reluLayer,layer4,layer5,output_layer]; % DNN structure

%% NN training
training_set_temp = training_set;
% training_set_temp = reshape(training_set,nTX*nRX*2,NN);
options = trainingOptions('adam', ...
    'MaxEpochs',15,...
    'InitialLearnRate',5*1e-3, ...
    'Verbose',true, ...
    'ValidationData',{training_set_temp(:,:,:,(NN/NN_fac+1):end),label_set((NN/NN_fac+1):end).'}, ...
    'ValidationFrequency',500,...
    'Plots','training-progress',...
    'Shuffle','every-epoch',...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.5,...
    'LearnRateDropPeriod',2);%,...
%     'MiniBatchSize',64);
trained_net = trainNetwork(training_set_temp(:,:,:,1:NN/NN_fac),label_set(1:NN/NN_fac).',layers,options);

%% verify distortion accuracy
disp('Checking accuracy')
CB_out = classify(trained_net,training_set_temp(:,:,:,(NN/NN_fac+1):end));
nn_c = 0;
NN_inn_p = zeros(length(CB_out),1);
for nn = (NN/NN_fac+1):NN
    nn_c = nn_c + 1;
    if ~mod(nn_c,100)
            nn_c
    end    
    U = training_set(:,:,1,nn) + 1i*training_set(:,:,2,nn);
    NN_inn_p(nn_c) = abs(trace(CB(:,:,CB_out(nn_c))'*(U*U')*CB(:,:,CB_out(nn_c))));
end
mean(NN_inn_p) % distortion of DNN
mean(inn_p_store) % distortion of exhaustive search

save(['Net_' file_name ],'trained_net','drop_prob','net_width');