function [ T ] = find_features( im, sigma )
%FIND_FEATURES returns the tensorfield after voting on the binary image im
%   using the sigma supplied.
%
%   T = find_features(im,sigma)
%
%   IM should be a binary (logical) image.
%   SIGMA should be the scale used in voting, i.e. 18.25.
%
%   Returns a tensor field T
%

    % Calculate cached voting field at various angles, this way we can save
    % a lot of time by preprocessing this data.
    cached_vtf = create_cached_vf(sigma);
    
    % normalize the gray scale image from 0 to 1
    im = double(im) / double(max(im(:)));
    
    % First step is to produce the initially encode the image
    % as sparse tensor tokens.
    sparse_tf = calc_sparse_field(im);
    
    % First run of tensor voting, use ball votes weighted by
    % the images grayscale.
    refined_tf = calc_refined_field(sparse_tf,im,sigma);
    
    % third run is to apply the stick tensor voting after
    % zero'ing out the e2(l2) components so that everything
    % is a stick vote.
    
    [e1,e2,l1,l2] = convert_tensor_ev(refined_tf);
    l2(:) = 0;
    zerol2_tf = convert_tensor_ev(e1,e2,l1,l2);
    
     
    T = calc_vote_stick(zerol2_tf,sigma,cached_vtf);
end