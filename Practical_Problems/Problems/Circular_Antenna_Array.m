function y = Circular_Antenna_Array(x)
    if size(x, 1) > size(x, 2), x = x'; end
    n = length(x);
    lb = get_xl(n)';
    ub = get_xu(n)';
    x = abs(ub - lb).*x + lb;
    y = Fitness(x);
end

function xl = get_xl(~)
    xl = [ones(6, 1)*0.2; -ones(6, 1)*180];
end

function xu = get_xu(~)
    xu = [ones(6, 1)*1; ones(6, 1)*180];
end

function y = Fitness(x)
    % Constants and parameters
    null = 50;
    phi_desired = 180;
    distance = 0.5;
    dim = length(x);
    phizero = 0;
    [~, num_null] = size(null);
    num1 = 300;
    phi = linspace(0, 360, num1);

    % Calculate array factor for different phi angles and find maximum
    yax(1) = array_factor(x, (pi/180)*phi(1), phi_desired, distance, dim);
    maxi = yax;
    phi_ref = 1;
    for i = 2:num1
        yax(i) = array_factor(x, (pi/180)*phi(i), phi_desired, distance, dim); %#ok<AGROW>
        if maxi < yax(i)
            maxi = yax(i);
            phizero = phi(i);
            phi_ref = i;
        end
    end

    % Find sidelobes
    count = 0;
    if yax(1) > yax(num1) && yax(1) > yax(2)
        count = count + 1;
        sidelobes(count) = yax(1);
        sllphi(count) = phi(1);
    end
    if yax(num1) > yax(1) && yax(num1) > yax(num1-1)
        count = count + 1;
        sidelobes(count) = yax(num1);
        sllphi(count) = phi(num1);
    end
    for i = 2:num1-1
        if yax(i) > yax(i+1) && yax(i) > yax(i-1)
            count = count + 1;
            sidelobes(count) = yax(i);
            sllphi(count) = phi(i);
        end
    end
    sidelobes = sort(sidelobes, 'descend');
    upper_bound = 180;
    lower_bound = 180;
    y = sidelobes(2)/maxi;
    sllreturn = 20*log10(y);

    % Calculate upper and lower beamwidth bounds
    for i = 1:num1/2
        if (phi_ref + i) > num1-1
            upper_bound = 180;
            break;
        end
        if yax(phi_ref + i) < yax(phi_ref + i - 1) && yax(phi_ref + i) < yax(phi_ref + i + 1)
            upper_bound = phi(phi_ref + i) - phi(phi_ref);
            break;
        end
    end

    for i = 1:num1/2
        if (phi_ref - i < 2)
            lower_bound = 180;
            break;
        end
        if yax(phi_ref - i) < yax(phi_ref - i - 1) && yax(phi_ref - i) < yax(phi_ref - i + 1)
            lower_bound = phi(phi_ref) - phi(phi_ref - i);
            break;
        end
    end
    bwfn = upper_bound + lower_bound;

    % Calculate the objective function components
    y1 = 0;
    for i = 1:num_null
        % The objective function for null control is calculated here
        y1 = y1 + (array_factor(x, null(i), phi_desired, distance, dim)/maxi);
    end
    
    y3 = abs(phizero - phi_desired);
    if y3 < 5
        y3 = 0;
    end

    y = 0;
    if bwfn > 80
        y = y + abs(bwfn - 80);
    end

    % Combine the components to calculate the final fitness value 'y'
    y = sllreturn + y + y1 + y3;

    % Check for NaN value and set a large value if necessary
    if isnan(y)
        y = 10^100;
    end
end

function y = array_factor(x1, phi, phi_desired, distance, dim)
    pi = 3.141592654;
    y = 0;
    y1 = 0;

    for i1 = 1:dim/2
        delphi = 2*pi*(i1-1)/dim;
        shi = cos(phi - delphi) - cos(phi_desired*(pi/180) - delphi);
        shi = shi * dim * distance;
        y = y + x1(i1) * cos(shi + x1(dim/2 + i1)*(pi/180));
    end

    for i1 = dim/2 + 1:dim
        delphi = 2*pi*(i1-1)/dim;
        shi = cos(phi - delphi) - cos(phi_desired*(pi/180) - delphi);
        shi = shi * dim * distance;
        y = y + x1(i1 - dim/2) * cos(shi - x1(i1)*(pi/180));
    end

    for i1 = 1:dim/2
        delphi = 2*pi*(i1-1)/dim;
        shi = cos(phi - delphi) - cos(phi_desired*(pi/180) - delphi);
        shi = shi * dim * distance;
        y1 = y1 + x1(i1) * sin(shi + x1(dim/2 + i1)*(pi/180));
    end

    for i1 = dim/2 + 1:dim
        delphi = 2*pi*(i1-1)/dim;
        shi = cos(phi - delphi) - cos(phi_desired*(pi/180) - delphi);
        shi = shi * dim * distance;
        y1 = y1 + x1(i1 - dim/2) * sin(shi - x1(i1)*(pi/180));
    end

    y = y * y + y1 * y1;
    y = sqrt(y);
end