function y = Spread_Spectrum_Radar(x)
    if size(x, 1) > size(x, 2), x = x'; end
    n = length(x);
    lb = get_xl(n)';
    ub = get_xu(n)';
    x = abs(ub - lb).*x + lb;
    
    % Determine the number of variables and initialize variables
    [~, d] = size(x);
    var = 2 * d - 1;
    hsum = zeros(var, 1);
    
    % Calculate the values of hsum vector
    for kk = 1:2 * var
        if rem(kk, 2) ~= 0
            % Odd index
            i = (kk + 1) / 2;
            hsum(kk) = 0;
            for j = i:d
                summ = 0;
                for i1 = (abs(2 * i - j - 1) + 1):j
                    summ = x(i1) + summ;
                end
                hsum(kk) = cos(summ) + hsum(kk);
            end
        else
            % Even index
            i = kk / 2;
            hsum(kk) = 0;
            for j = (i + 1):d
                summ = 0;
                for i1 = (abs(2 * i - j) + 1):j
                    summ = x(i1) + summ;
                end
                hsum(kk) = cos(summ) + hsum(kk);
            end
            hsum(kk) = hsum(kk) + 0.5;
        end
    end
    
    % Calculate the maximum value of hsum as the fitness value y
    y = max(hsum);
    
    % Check for NaN value and set a large value if necessary
    if isnan(y)
        y = 10^100;
    end
end

function xl = get_xl(nx)
    xl = zeros(nx, 1);
end

function xu = get_xu(nx)
    xu = ones(nx, 1).*2*pi;
end