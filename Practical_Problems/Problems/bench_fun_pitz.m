function [out] = bench_fun_pitz(x)

if size(x, 2) > size(x, 1), x = x'; end
dim = length(x);
xL = get_xl(dim);
xU = get_xu(dim);
x = abs(xU - xL).*x + xL;


bounds = [xL,xU];

[m,n] = size(x);

% problem bounds
lb = [0.0063   -0.0465    0.0019   -0.0484    0.0167   -0.0116    0.0046   -0.0439    0.0034   -0.0471];
ub = [0.2339    0.0093    0.1333    0.0078    0.2300    0.0103    0.2267    0.0046    0.2408    0.0100];

if m == 1
    y = change_bounds(x,bounds,lb,ub);
    out = fhd(y);
    return;
end

if m == dim
    out = zeros(n,1);
    for i=1:n
        y = change_bounds(x(:,i)',bounds,lb,ub);
        out(i) = fhd(y);
    end
    return;
else
    out = zeros(m,1);
    for i=1:m
        y = change_bounds(x(i,:),bounds,lb,ub);
        out(i) = fhd(y);
    end
    return
end

end

% running the docker/objective function evaluation
function out = fhd(x)
    %str = strcat("docker run --rm frehbach/cfd-test-problem-suite ./dockerCall.sh PitzDaily '",sprintf('%4.3f,',x),sprintf('\b'),"'");
    str = strcat("docker run --rm frehbach/cfd-test-problem-suite ./dockerCall.sh PitzDaily ",sprintf('%4.3f,',x),sprintf('\b'),"");
    [~,val] = system(str);
    if strcmp(val(1:end-1),'Error')
        out = 1;
    else
        out = str2num(val);
    end
end

% function for recomputing the bounds
function y = change_bounds(x,bounds,lb,ub)
    dim = length(x);
    range = bounds(1,2)-bounds(1,1);
    y = 0*x;
    for i=1:dim
        y(i) = x(i)/range.*(ub(i)-lb(i))+(lb(i)+ub(i))/2;
    end
end

function xl = get_xl(nx)
    xl = -10*ones(nx,1);
end

function xu = get_xu(nx)
    xu = 10*ones(nx,1);
end