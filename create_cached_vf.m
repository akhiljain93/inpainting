function [ out ] = create_cached_vf( sigma )
	ws = floor( ceil(sqrt(-log(0.01)*sigma^2)*2) / 2 )*2 + 1;  
    out = zeros(180,ws,ws,2,2);
    for i=1:180
        x = cos(pi/180*i);
        y = sin(pi/180*i);
        v = [x,y];
        Fk = create_stick_tensorfield(v,sigma);
        out(i,:,:,:,:) = Fk;
    end

end