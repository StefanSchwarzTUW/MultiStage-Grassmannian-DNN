%% Quantization of Time-Correlated Channels with Selective Stage Update according to "Recursive CSI Quantization of Time-Correlated MIMO Channels by Deep Learning Classification", S. Schwarz, IEEE SPL 2020
% (c) Stefan Schwarz @ Institute of Telecommunications, TU Wien 2020

% clc;
% clear all;
% close all;
% warning off;

% Nt = 4;
% Nr = 2;
NN = 1500; % number of TTIs
OO = 10;  % number of random realizations

bits_dim = log2(CB_size); % bits per dimension
fd = 10; % Doppler frequency unnormalized (normalized is fd*1e-3, due to 1ms TTI duration)

% hysteresis parameters of the selective stage update
hyst_low = 1.05; 
hyst_high = 1.025;

use_NN = true; % (de)active DNN classification

dim_vec = [Nt:-1:Nr];
Ndim = length(dim_vec)-1;
dim_diff = ones(Ndim,1); % dimensionality stepsize

% required for calculation of theoretical average distortion of the individual stages
dist_fac = zeros(1,Ndim);
for nt = 1:Ndim
    p = Nr;
    n = dim_vec(nt);
    q = dim_vec(nt+1);
    c1 = 1/gamma(p*(n-q)+1);
    for i = 1:p
        c1 = c1*gamma(n-i+1)/gamma(q-i+1);
    end
    pre_factor1 = gamma(1/(p*(n-q)))/(p*(n-q));
    K1 = (c1)^(-1/(p*(n-q)));
    dist_fac(nt) = pre_factor1*K1;
end

w_d = 2*pi*fd; % Doppler frequency in [rad]
alpha = besselj(0,w_d*10^-3); % correlation factor for Gauss-Markov model

bits_vec = bits_dim*ones(Ndim,1);

% theoretical average distortion of the individual stages
dc_theor = zeros(Ndim,1);
for nt = 1:Ndim
    p = Nr;
    n = dim_vec(nt);
    q = dim_vec(nt+1);
    K1 = (2^(bits_vec(nt)))^(-1/(p*(n-q)));       
    if bits_vec(nt) <= 1 && p == 1
        dc_temp1 = 1-q/n; % random isotropic projection in case of 0bit codebook
        dc_temp2 =  dist_fac(nt)*K1/Nr;
        prob = max((2^(bits_vec(nt))-max(floor(2^(bits_vec(nt))),1)),0);
        dc_theor(nt) = (1-prob)*dc_temp1 + prob*dc_temp2;
    else
        dc_theor(nt) = dist_fac(nt)*K1/Nr;    % theoretic normalized chordal distance (normalized by Nr)
    end
end
dc_theor_full = 1-prod(1-dc_theor); % theoretical distortion of multi stage quantization
reverse_prod = cumprod(1-dc_theor(end:-1:1));
reverse_prod = reverse_prod(end:-1:1); % required for stage update

% theoretical distortion of single stage quantization
p = Nr;
n = Nt;
q = Nr;
c1 = 1/gamma(p*(n-q)+1);
for i = 1:p
    c1 = c1*gamma(n-i+1)/gamma(q-i+1);
end
pre_factor1 = gamma(1/(p*(n-q)))/(p*(n-q));
K1 = (c1*2^(sum(bits_vec)))^(-1/(p*(n-q)));
dc_theor_single = pre_factor1*K1/Nr;

inn_prod = zeros(NN,OO);
inn_prod_scalar = zeros(NN,OO);
bits_store = zeros(OO,1);
ind_store = zeros(NN,OO);

for oo = 1:OO
    oo
s_temp = RandStream('mt19937ar','Seed',oo+5);
H = 1/sqrt(2)*(randn(s_temp,Nt,Nr)+1i*randn(s_temp,Nt,Nr)); % Rayleigh fading channel
initial_phase = true;
nn_c = 0;
CB_rec = cell(Ndim,1);


net = cell(Ndim,1);
ind_store(1,oo) = Ndim;
bits_acc = 0;
for nn = 1:NN
    H = alpha*H + sqrt(1-alpha^2)*(1/sqrt(2)*(randn(s_temp,Nt,Nr)+1i*randn(s_temp,Nt,Nr)));  % Gauss-Markov channel
    [U,~,~] = svd(H,'econ'); % subspace to be quantized
    if ~mod(nn-1,50) % update the RVQ codebook every now and then; random codebook generation
        nn
        if use_NN && nn == 1 % for the DNN we don't need a RVQ update
            for cb_i = 1:Ndim
                CB_size = ceil(2.^bits_vec(cb_i));
                file_name = [num2str(Nt-cb_i+1) '_' num2str(Nr) '_' num2str(log2(CB_size)) '_' num2str(6) '.mat'];
                load([file_name]);
                CB_rec{cb_i} = CB;
                load(['Net_' file_name]);
                net{cb_i} = trained_net;                
            end
            if oo == 1
                disp('Notice: in Matlab the DNN implementation is not fast, since loading/initialization of the net of each stages takes too long')
            end
        elseif ~use_NN
            for cb_i = 1:Ndim
                CB_size = ceil(2.^bits_vec(cb_i));
                dim1 = dim_vec(cb_i);
                quant_dim = dim_diff(cb_i);
                CB_rec{cb_i} = RANDOM_MIMO_CB(quant_dim,dim1,CB_size,s_temp,0,1);  
            end
        end
    end
    if initial_phase % initialization requires full quantization of all stages
        B = U;
        Ht = 1;
        inn_store = cell(Ndim,1); % inner product of quantization stages
        CB_store = cell(Ndim,1); % quantized codebook entry
        for cc = 1:Ndim
            if use_NN % quantization using DNN
                [Uh,B,bits,CB_entry] = quant_NN(B,net{cc},CB_rec{cc},dim_vec(cc+1));
            else % quantization using exhaustive search
                [Uh,B,bits,dc,CB_entry] = quant_Grass(B,s_temp,bits_vec(cc),dim_vec(cc+1),CB_rec{cc});    
            end
            bits_acc = bits_acc + bits; % accumulate the number of quantization bits
            Ht = Ht*Uh;    % CSI reconstruction    
            inn_store{cc} = real(trace(U'*(Ht*Ht')*U))/Nr;
            CB_store{cc} = CB_entry;
        end
        inn_prod(nn,oo) = real(trace(U'*(Ht*Ht')*U))/Nr;        
        initial_phase = false;
    else
        Ht = 1;
        inn_temp = zeros(Ndim,1);
        for cc = 1:Ndim 
            [Q,~,~] = svd(CB_store{cc}); % CB is the 1-D null-space (quantization in the null-space)
            Uh = Q(:,2:end); 
            Ht = Ht*Uh; % reconstructed CSI of the individual stages
            inn_temp(cc) = real(trace(U'*(Ht*Ht')*U))/Nr; % inner product with current channel subspace
        end
        inn_prod(nn,oo) = inn_temp(end); 
        if inn_temp(end) < (1-hyst_low*dc_theor_full)  % if inner product not sufficiently high --> update some stages         
            inn_vals = [reverse_prod(1);inn_temp(1:end-1).*reverse_prod(2:end)]; % expected inner product when updating [all stages,stages 2:end, stages 3:end, ... , last stage] 
            [ind,~] = find(inn_vals > (1-hyst_high*dc_theor_full),1,'last'); % determine how many stages we need to update
            ind_store(nn,oo) = Ndim-ind+1; % store number of stages that have to be updated            
            Ht = 1;   
            B = U; % current full CSI
            if ind > 1  % calculate old CSI of stages that will not be updated
                for cc = 1:ind-1
                    [Q,~,~] = svd(CB_store{cc}); % CB is the 1-D null-space (quantization in the null-space)
                    Uh = Q(:,2:end); 
                    Ht = Ht*Uh; % reconstruct CSI of not updated stages
                    [W,L,~] = svd(B'*(Uh*Uh')*B); % update the SQBC matrix according to the current CSI, given the fixed CB values of the not updated quantizer stages
                    B = Uh'*B*W*L^(-1/2);
                end
            end
            for cc = ind:Ndim % update the later stages of the quantizer
                if use_NN
                    [Uh,B,bits,CB_entry] = quant_NN(B,net{cc},CB_rec{cc},dim_vec(cc+1));
                else
                    [Uh,B,bits,dc,CB_entry] = quant_Grass(B,s_temp,bits_vec(cc),dim_vec(cc+1),CB_rec{cc});   
                end
                bits_acc = bits_acc + bits;
                Ht = Ht*Uh;    % CSI reconstruction
                inn_store{cc} = real(trace(U'*(Ht*Ht')*U))/Nr;
                CB_store{cc} = CB_entry;
            end
            % CSI reconstruction at the TX from codebook entries only (to make sure that everything can be reconstructed from these values)
            Ht = 1;
            for cc = 1:Ndim
                [Q,~,~] = svd(CB_store{cc}); % CB is the 1-D null-space 
                Uh = Q(:,2:end); 
                Ht = Ht*Uh;
            end
            inn_prod(nn,oo) = real(trace(U'*(Ht*Ht')*U))/Nr;   
        end
    end
    %% scalar quantization (magnitude and phase)
%     Ut = U*exp(-1i*angle(U(1,1)));
%     phase_inds = discretize(angle(Ut),phase_bins);
%     mag_inds = discretize(abs(Ut),mag_bins);
%     [Uh,~,~] = svd(mag_vals(mag_inds).*exp(1i*phase_vals(phase_inds)),'econ');
%     inn_prod_scalar(nn) = abs(trace(Uh'*(Ut*Ut')*Uh));
    
end
bits_store(oo) = bits_acc/NN;
end
(1-mean(mean(inn_prod)))
mean(bits_store)

figure(1)
[N,X] = hist(ind_store(:),[0:Ndim]);
bar(X,N/sum(N));
hold on
xlabel('Number of updated stages')
ylabel('Frequency of update')
figure(2)
semilogy((1-mean(inn_prod,2)))
hold on
grid on
xlabel('Normalized quantization error over TTIs')
ylabel('Chordal distance error')
% semilogy(1-inn_prod_scalar)

save(fullfile(['Quant_' num2str(Nt) '_' num2str(Nr) '_' num2str(bits_dim) '_' num2str(fd) '_' num2str(use_NN) '.mat']))
