function [out] = bench_fun_eps(x)

if size(x, 2) > size(x, 1), x = x'; end
dim = length(x);
xL = get_xl(dim);
xU = get_xu(dim);
x = abs(xU - xL).*x + xL;


bounds = [xL,xU];
[m,n] = size(x);

% problem bounds
lb = zeros(1,dim);
ub = 7.999*ones(1,dim);

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
    str = strcat("docker run --rm frehbach/cfd-test-problem-suite ./dockerCall.sh ESP '",sprintf('%i,',x),sprintf('\b'),"'");
    [~,val] = system(str);
    if strcmp(val(1:end-1),'Error')
        out = 1;
    else
        out = str2num(val);
        out = out(end);
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
    y = floor(y);
end

function xl = get_xl(nx)
    xl = -10*ones(nx,1);
end

function xu = get_xu(nx)
    xu = 10*ones(nx,1);
end