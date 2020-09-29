%% Neural Network Training Data for "Recursive CSI Quantization of Time-Correlated MIMO Channels by Deep Learning Classification", S. Schwarz, IEEE SPL 2020
% (c) Stefan Schwarz @ Institute of Telecommunications, TU Wien 2020
% Small-scale example to speed-up DNN training and quantization

clc;
clear all;
close all;

Nt = 4;   % number of transmit antennas (n)
Nr = 2;    % number of receive antennas (m)
CB_size = 2^6;  % quantization codebook size
d_vec = Nt:-1:Nr;
train_network = true;

if train_network % activate this if the DNN has to be trained
    for di = 1:length(d_vec)-1 % DNNs for the individual stages of the quantizer
        disp(di);
        nTX = d_vec(di);
        nRX = Nr;
        NN_training; % generate training data
        train_net; % train the network
    end
end

time_corr; % apply the quantizer for quantization of a time-correlated channel

