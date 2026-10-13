function y = WindWake(x)
persistent fhd
if isempty(fhd)
    n = length(x)/2;
    fhd = pyrunfile('windwake.py', "prob", n_turbines = n, wind_seed = 0, n_samples = 5);
end
if size(x, 2) > size(x, 1), x = x'; end

y = fhd.evaluate(x);
return