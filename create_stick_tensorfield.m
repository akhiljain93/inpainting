function [ T ] = create_stick_tensorfield( uv, sigma )
%CREATE_STICK_TENSORFIELD Creates a second order tensor 
%   field aligned along the unit vector provided and with 
%   the scale of sigma.
%
%   unit_vector describes the direction the field should go in
%   it's a 2 column, 1 row matrix where the first element is
%   the x axis, the second element is the y axis. Default is
%   [1,0], aligned on the x-axis.
%
%   sigma describes the scale for the tensorfield, default is
%   18.25.
%
%   Ret urns T a MxMx2x2 tensor field, where M is the rectangular
%   size of the field.
%
%   Example:
%       T = create_stick_tensorfield(unit_vector, sigma);
%

    % Generate the defaults if they're not available.
        if nargin < 1
            uv = [1,0];
        end
        if nargin < 2
            sigma = 18.25;
        end
    
    % Generate initial parameters used by the entire system

        % Calculate the window size from sigma using 
        % equation 5.7 from Emerging Topics in Computer Vision
        % make the field odd, if it turns out to be even.
        ws = floor( ceil(sqrt(-log(0.01)*sigma^2)*2) / 2 )*2 + 1;
        whalf = (ws-1)/2;

        % Turn the unit vector into a rotation matrix
        rot = [uv(:),[-uv(2);uv(1)]]/norm(uv);
        btheta = atan2(uv(2),uv(1));

        % Generate our theta's at each point in the
        % field, adjust by our base theta so we rotate
        % in funcion.
        [X,Y] = meshgrid(-whalf:1:whalf,whalf:-1:-whalf);
        Z = rot'*[X(:),Y(:)]';
        X = reshape( Z(1,:),ws,ws);
        Y = reshape( Z(2,:),ws,ws);
        theta = atan2(Y,X);

    % Generate the tensor field direction aligned with the normal
        Tb = reshape([theta,theta,theta,theta],ws,ws,2,2);
        T1 = -sin(2*Tb+btheta);
        T2 = cos(2*Tb+btheta);
        T3 = T1;
        T4 = T2;
        T1(:,:,2,1:2) = 1;
        T2(:,:,1:2,1) = 1;
        T3(:,:,1:2,2) = 1;
        T4(:,:,1,1:2) = 1;
        T = T1.*T2.*T3.*T4;
        
        
    % Generate the attenuation field, taken from Equation
    % 5.2 in Emerging Topics in Computer Vision. Note our
    % thetas must be symmetric over the Y axis for the arc
    % length to be correct so there's a bit of a coordinate
    % translation.
        theta = abs(theta);
        theta(theta>pi/2) = pi - theta(theta>pi/2);
        theta = 4*theta;
        
        s = zeros(ws,ws);
        k = zeros(ws,ws);
        % Calculate the attenuation field.
        l = sqrt(X.^2+Y.^2);
        c = (-16*log2(0.1)*(sigma-1))/pi^2;
        s(l~=0 & theta~=0) = (theta(l~=0 & theta~=0).*l(l~=0 & theta~=0))./sin(theta(l~=0 & theta~=0));
        s(l==0 | theta==0) = l(l==0 | theta==0);
        k(l~=0) = 2*sin(theta(l~=0))./l(l~=0);
        DF = exp(-((s.^2+c*(k.^2))/sigma^2));
        DF(theta>pi/2) = 0;
    % Generate the final tensor field
        T = T.*reshape([DF,DF,DF,DF],ws,ws,2,2);
   
end