function [ T ] = calc_refined_field( tf, im, sigma )
    
    % Get votes for ball
    ball_vf = calc_vote_ball(tf,im,sigma);
    
    % Erase anything that's not in the original image
    [rows cols] = find(im==0);
    s = size(rows,1);
    
    for i=1:s
        ball_vf(rows(i),cols(i),1,1) = 0;
        ball_vf(rows(i),cols(i),1,2) = 0;
        ball_vf(rows(i),cols(i),2,1) = 0;
        ball_vf(rows(i),cols(i),2,2) = 0;
    end
    
    T = tf + ball_vf;
end
    
    