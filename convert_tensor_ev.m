function [ o1, o2, o3, o4 ] = convert_tensor_ev( i1, i2, i3, i4 )
%CONVERT_TENSOR_EV converts the tensor field to eigenvectors and
%   eigenvalues, vise versa.
%
%   [E1,E2,L1,L2] = CONVERT_TENSOR_EV(TENSOR_FIELD) converts a
%   tensor field to eigenvectors and eigenvalues.
%   
%   TENSOR_FIELD = CONVERT_TENSOR_EV(E1,E2,L1,L2) converts
%   eigenvectors and eigenvalues to a tensor field.
%
%   Example:
%   [e1,e2,l1,l2] = convert_tensor_ev(tensor_field)
%
%   tensor_field = conver_tensor_ev(e1,e2,l1,l2);
%
    if nargin==1
        K11 = i1(:,:,1,1);
        K12 = i1(:,:,1,2);
        K21 = i1(:,:,2,1);
        K22 = i1(:,:,2,2);

        [n,p] = size(K11);

        o1 = zeros(n,p,2);
        o2 = zeros(n,p,2);
        o3 = zeros(n,p);
        o4 = zeros(n,p);

        % trace/2
        t = (K11+K22)/2;

        a = K11 - t;
        b = K12;

        ab2 = sqrt(a.^2+b.^2);
        o3 = ab2  + t;
        o4 = -ab2 + t;

        theta = atan2( ab2-a, b );

        o1(:,:,1) = cos(theta);
        o1(:,:,2) = sin(theta);
        o2(:,:,1) = -sin(theta); 
        o2(:,:,2) = cos(theta);
    else
        o1 = zeros( [size(i3),2,2] );
        o1(:,:,1,1) = i3.*i1(:,:,1).^2 + i4.*i2(:,:,1).^2;
        o1(:,:,1,2) = i3.*i1(:,:,1).*i1(:,:,2) + i4.*i2(:,:,1).*i2(:,:,2);
        o1(:,:,2,1) = o1(:,:,1,2);
        o1(:,:,2,2) = i3.*i1(:,:,2).^2 + i4.*i2(:,:,2).^2;
    end



end