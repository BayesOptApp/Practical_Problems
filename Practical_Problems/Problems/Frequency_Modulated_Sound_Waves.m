function y = Frequency_Modulated_Sound_Waves(x)
    if size(x, 2) > size(x, 1), x = x'; end
    n = length(x);
    lb = get_xl(n);
    ub = get_xu(n);
    x = abs(ub - lb).*x + lb;

    % Define the value of theta
    theta = 2 * pi / 10;
    
    % Initialize the output value y
    y = 0;
    
    % Determine the number of groups 'g' based on the length of x
    g = length(x) / 6;
    
    % Iterate over each group 'j'
    for j = 1:g
        % Iterate over 't' from 1 to 10
        for t = 1:10
            y_t = x(1 + 6 * (j - 1)) * sin(x(2 + 6 * (j - 1)) * t * theta + x(3 + 6 * (j - 1)) * sin(x(4 + 6 * (j - 1)) * t * theta + x(5 + 6 * (j - 1)) * sin(x(6 + 6 * (j - 1)) * t * theta)));
            y_0_t = 1 * sin(5 * t * theta - 1.5 * sin(4.8 * t * theta + 2 * sin(4.9 * t * theta)));
            y = y + (y_t - y_0_t)^2;
        end
    end
end

function xl = get_xl(nx)
    xl = -ones(nx, 1).*6.4;
end

function xu = get_xu(nx)
    xu = ones(nx, 1).*6.35;
end