function T = calc_vote_stick(T,sigma,cachedvf)
%CALC_VOTE_STICK votes using stick tensors returning a new
%   tensor field.
%
%   T = calc_vote_stick(M,sigma,cachedvf);
%
%   M is the input tensor field, this should be an estimate.
%   sigma determines the scale of the tensor field.
%   cachedvf is a cached voting field produced by
%   create_cached_vf in order to speed up voting process.
%   
    
    if nargin<2
        sigma = 18.25;
    end

    wsize = floor( ceil(sqrt(-log(0.01)*sigma^2)*2) / 2 )*2 + 1;
    wsize_half = (wsize-1)/2;
    
    % resize the tensor to make calculations easier. This gives us a margin
    % around the tensor that's as large as half the window of the voting
    % field so when we begin to multiple and add the tensors we dont have
    % to worry about trimming the voting field to avoid negative and
    % overflow array indices.
    Th = size(T,1);
    Tw = size(T,2);
    
    Tn = zeros(Th+wsize_half*2,Tw+wsize_half*2,2,2,'double');
    Tn((wsize_half+1):(wsize_half+Th), (wsize_half+1):(wsize_half+Tw), :, :) = T(1:end,1:end,:,:);
    T = Tn;
    
    % perform eigen-decomposition, assign default tensor from estimate
    [e1,e2,l1,l2] = convert_tensor_ev(T);
    
    % Find everything that's a stick vote.  Prehaps use a threshold here?
    I = find ( l1-l2 > 0 );
    
    % perform stick voting
    d = waitbar(0,'Please wait, stick voting...');
    
    % Loop through each stick found in the tensor T.
    a = 0;

    [u,v] = ind2sub(size(l1),I');
    p = size(u,2);
    D = zeros(2,p,'double');
    D(1,:) = u;
    D(2,:) = v;
    op = ceil(p*0.01);
    
    for s = D;
        a = a+1;
        
        if mod(a,op) == 0
            waitbar(a/p,d);
        end

        % the direction is e1 with intensity l1-l2
        v = e1(s(1),s(2),:);
        
        if nargin < 6
            %v(:,:,1) = -v(:,:,1);
            v=v(:);
            Fk = create_stick_tensorfield([-v(2),v(1)],sigma);
        else
            angle = round(180/pi*atan(v(2)/v(1)));
            if angle < 1
                angle = angle + 180;
            end
            Fk = shiftdim(cachedvf(angle,:,:,:,:));
        end
       
        % the current intensity of the vote
        % Apply weights
        Fk = (l1(s(1),s(2))-l2(s(1),s(2)))*Fk;
        
        % Calculate positions of window, add subsequent values back in.
        beginy = s(1)-wsize_half;
        endy = s(1)+wsize_half;
        beginx = s(2)-wsize_half;
        endx = s(2)+wsize_half;
        T(beginy:endy,beginx:endx,:,:) = T(beginy:endy,beginx:endx,:,:) + Fk;
    end
    
    close(d);
    % Trim the T of the margins we used.
    T = T((wsize_half+1):(wsize_half+Th), (wsize_half+1):(wsize_half+Tw), :, :);
    
end