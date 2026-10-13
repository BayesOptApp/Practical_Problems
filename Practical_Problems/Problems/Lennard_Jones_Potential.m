function y = Lennard_Jones_Potential(x)
    if size(x, 1) > size(x, 2), x = x'; end
    n = length(x);
    lb = get_xl(n)';
    ub = get_xu(n)';
    x = abs(ub - lb).*x + lb;
    p = size(x);
    n = p(2)/3;
    x = reshape(x, 3, n)';
    r = ones(3, n);
    y = 0;
    a = ones(n, n);
    b = repmat(2,n,n);
    for i = 1:(n - 1)
        for j = (i + 1):n
            r(i, j) = max(sqrt(sum((x(i, :) - x(j, :)).^2)), 1e-16); % fixed
            y = y + (a(i, j)/r(i, j)^12 - b(i, j)/r(i, j)^6);
        end
    end
end

function xl = get_xl(nx)
    xl = zeros(3, 1);
    for i = 4:nx
        xl(end + 1) = -4 - (0.25)*((i-4)/3); %#ok<*AGROW>
    end
end

function xu = get_xu(nx)
    xu = [4; 4; 3];
    for i = 4:nx
        xu(end + 1) = 4 + (0.25)*((i-4)/3);
    end
end