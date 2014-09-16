function [ T ] = create_ball_tensorfield( sigma )
%CREATE_BALL_TENSORFIELD creates a ball tensor field, sigma
%   determines the scale and size of the tensor field.
%
%   Default for sigma is 18.25.

    if nargin < 1
        sigma = 18.25;
    end

    wsize = ceil(sqrt(-log(0.01)*sigma^2)*2);
    wsize = floor(wsize/2)*2+1;
    
    T = zeros(wsize,wsize,2,2);
    for theta = (0:1/32:1-1/32)*2*pi
        v = [cos(theta);sin(theta)];
        B = create_stick_tensorfield(v,sigma);
        T = T + B;
    end
    T = T/32;
end