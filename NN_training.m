%% Neural Network Training Data for "Recursive CSI Quantization of Time-Correlated MIMO Channels by Deep Learning Classification", S. Schwarz, IEEE SPL 2020
% (c) Stefan Schwarz @ Institute of Telecommunications, TU Wien 2020

% clc;
% clear all;
% close all;
% 
% nTX = 4;   % number of transmit antennas (n)
% nRX = 2;    % number of receive antennas (m)
% CB_size = 2^6;  % quantization codebook size
seed = RandStream('mt19937ar','Seed',20); 
NN = 1e6; % number of training samples
RR = 1e3; % number of generated codebooks -- out of these codebooks, the one that minimizes the average distortion is used for training
%% Generation of RVQ codebook
inn_p_temp = nRX;
for rr = 1:RR % from RR random realizations, pick the one with minimum average distortion
    if ~mod(rr,100)
            rr
    end
    CB_temp = RANDOM_MIMO_CB(1,nTX,CB_size,seed,0,1); 
    inn_p_vec = [];
    for cb_i1 = 1:CB_size
        for cb_i2 = cb_i1+1:CB_size
            inn_p_vec = [inn_p_vec, abs(trace(CB_temp(:,:,cb_i1)'*(CB_temp(:,:,cb_i2)*CB_temp(:,:,cb_i2)')*CB_temp(:,:,cb_i1)))]; % null-space codebook
        end
    end
    if mean(inn_p_vec) < inn_p_temp
        CB = CB_temp;
        inn_p_temp = mean(inn_p_vec);
        rr_store = rr;
    end
end

%% Generation of the training set
training_set = zeros(nTX,nRX,2,NN);
% training_set = zeros(nTX*2,nRX,1,NN);
label_set = zeros(NN,1);
file_name = [num2str(nTX) '_' num2str(nRX) '_' num2str(log2(CB_size)) '_' num2str(log10(NN)) '.mat'];
% categories = num2cell(1:CB_size);
if ~exist(file_name,'file')
    inn_p_store = zeros(NN,1);
    parfor nn = 1:NN
        if ~mod(nn,100)
            nn
        end
        H = randn(seed,nTX,nRX) + 1i*randn(seed,nTX,nRX); % iid Rayleigh fading channel
        [U,~,~] = svd(H,'econ');
        U = U*diag(exp(-1i*angle(U(1,:)))); % phase rotation to make the first row real-valued
        training_set(:,:,:,nn) = cat(3,real(U),imag(U));
%         training_set(:,:,:,nn) = [abs(U).^2;angle(U)];
        inn_p = zeros(CB_size,1);
        for cb_i = 1:CB_size
            inn_p(cb_i) = abs(trace(CB(:,:,cb_i)'*(U*U')*CB(:,:,cb_i)));  % one minus chordal distance
        end
        [inn_p_store(nn),min_ind] = min(inn_p); % quantization in the null-space --> minimizing the overlap with the null-space codebook (see: "Reduced Complexity Recursive Grassmannian Quantization")
        label_set(nn) = min_ind; 
    end
    label_set = categorical(label_set,1:CB_size);
    save([num2str(nTX) '_' num2str(nRX) '_' num2str(log2(CB_size)) '_' num2str(log10(NN)) '.mat'],'training_set','label_set','CB','inn_p_store')
else
    disp('Data already exists')
end
