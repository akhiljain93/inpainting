function T = calc_vote_ball(T,im,sigma)

    Fk = create_ball_tensorfield(sigma);
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
    
    % Find everything that's above 0
    I = find ( l1 > 0 );
    
    % perform ball voting
    d = waitbar(0,'Please wait, ball voting...');
    
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
        
        % the current intensity of the vote
        % Apply weights
        Zk = im(s(1)-wsize_half,s(2)-wsize_half)*Fk;
        
        % Calculate positions of window, add subsequent values back in.
        beginy = s(1)-wsize_half;
        endy = s(1)+wsize_half;
        beginx = s(2)-wsize_half;
        endx = s(2)+wsize_half;
        T(beginy:endy,beginx:endx,:,:) = T(beginy:endy,beginx:endx,:,:) + Zk;
    end
    
    close(d);
    % Trim the T of the margins we used.
    T = T((wsize_half+1):(wsize_half+Th), (wsize_half+1):(wsize_half+Tw), :, :);
end