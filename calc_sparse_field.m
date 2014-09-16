function [ T ] = calc_sparse_field( image )
    [h w] = size(image);
    T = zeros(h,w,2,2);
    
    [rows,cols] = find(image>0);
    
    n = size(rows,1);
    for i=1:n
        T(rows(i),cols(i),:,:) = [1,0;0,1];
    end
end