% codebook of either iid Gaussian matrices or of orthonormal bases
% (c) Stefan Schwarz @ ICT, 2014
function CB = RANDOM_MIMO_CB(nRX,nTX,CB_size,seed,gauss,gains,varargin)
    CB = zeros(nTX,nRX,CB_size);
    if isempty(varargin)
        for ii = 1:CB_size        
            if gauss      
                CB(:,:,ii) = 1/sqrt(2)*(randn(seed,nTX,nRX)+1i*randn(seed,nTX,nRX));    % codebook of iid Gaussian matrices
            else
                H = diag(gains,0)*1/sqrt(2)*(randn(seed,nTX,nRX)+1i*randn(seed,nTX,nRX)); % "gains" accounts for channel gain differences between different transmit antennas
                % useful for matching the codebook to pathloss differences in distributed antenna systems (see "Single-user MIMO versus multi-user MIMO in distributed antenna systems with limited feedback" S.Schwarz, R.W.Heath, M.Rupp)
                [U,~,~] = svd(H,'econ');
                CB(:,:,ii) = U; % codebook of orthonormal bases           
            end
        end 
    else
        NS_proj = (eye(nTX)-varargin{1}*varargin{1}');
        for ii = 1:CB_size
            H = diag(gains,0)*1/sqrt(2)*(randn(seed,nTX,nRX)+1i*randn(seed,nTX,nRX)); % "gains" accounts for channel gain differences between different transmit antennas
            % useful for matching the codebook to pathloss differences in distributed antenna systems (see "Single-user MIMO versus multi-user MIMO in distributed antenna systems with limited feedback" S.Schwarz, R.W.Heath, M.Rupp)
            [U,~,~] = svd(H,'econ');
            CB(:,:,ii) = NS_proj*U; % codebook of orthonormal bases
            CB(:,:,ii) = CB(:,:,ii)/norm(CB(:,:,ii));
        end
    end
end