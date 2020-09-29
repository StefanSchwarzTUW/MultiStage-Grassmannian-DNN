function [Uq,B,bits,dc,CB_val,max_ind] = quant_Grass(U,s_temp,bits,quant_dim,CB_rec)
    dim1 = size(U,1);
    dim2 = size(U,2);

    if dim1 > 1
        CB_size_high = max(ceil(2^bits),1);
        CB_size_low = max(floor(2^bits),1);
        prob = 2^bits - CB_size_low;
        bern = rand(1) <= prob;
        CB_size = CB_size_low*(1-bern) + CB_size_high*bern; 
        
       CB = CB_rec(:,:,1:CB_size); 
%        rr = randperm(size(CB_rec,3));
%        rr1 = randperm(size(CB_rec,1));       
%        CB = CB_rec(rr1(1:dim1),1:(dim1-quant_dim),rr(1:CB_size));
%        CB = CB./repmat(sqrt(sum(abs(CB).^2,1)),dim1,1,1);
        
%         fft_size = max(CB_size*quant_dim,dim1);
%         CB = dftmtx(fft_size)*1/sqrt(dim1);
%         CB = CB(randperm(fft_size),randperm(fft_size));
%         CB = CB(1:dim1,1:CB_size*quant_dim);
%         CB = reshape(CB,dim1,quant_dim,CB_size);
        
        quante = zeros(size(CB,3),1);
        for b_i = 1:size(CB,3)
            temp = U'*CB(:,:,b_i);
            quante(b_i) = trace(temp*temp');
        end
        [max_quant,max_ind] = min(quante);
        [Uq,~,~] = svd(CB(:,:,max_ind));
        Uq = Uq(:,(dim1-quant_dim+1):end);
        CB_val = CB(:,:,max_ind);
        
%         [Uc,Sc,Vc] = svd(Uq,'econ');
        [W,L,~] = svd(U'*(Uq*Uq')*U);
        B = Uq'*U*W*L^(-1/2);   
        
%         B = (Uq'*U)*1/sqrt(max_quant);
        
%         trace((Uq*B)'*U*U'*(Uq*B))
%         bits = log2(nchoosek(dim1,n_fb_beams));
        dc = max_quant/dim2; % normalized chordal distance
        bits = log2(size(CB,3));
    else
        Uq = 1;
        CB_val = 1;
        B = 1;
        bits = 0;
        dc = 0;
        max_ind = 1;
    end
end