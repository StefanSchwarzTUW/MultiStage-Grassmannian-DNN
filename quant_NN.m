function [Uq,B,bits,CB_val] = quant_NN(U,net,CB,quant_dim)
    dim1 = size(U,1);
%     dim2 = size(U,2);
    bits = log2(size(CB,3));
    U = U*diag(exp(-1i*angle(U(1,:))));
    max_ind = classify(net,cat(3,real(U),imag(U)));
    [Uq,~,~] = svd(CB(:,:,max_ind));
    Uq = Uq(:,(dim1-quant_dim+1):end);
    CB_val = CB(:,:,max_ind);
    [W,L,~] = svd(U'*(Uq*Uq')*U);
    B = Uq'*U*W*L^(-1/2);
end