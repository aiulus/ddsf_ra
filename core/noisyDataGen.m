function [perturbed, nominal, u] = noisyDataGen(lookup)
    %% Extract parameters
    sys = lookup.sys;
    A = sys.A;
    B = sys.B;
    C = sys.C;
    D = sys.D;

    W = lookup.sys.params.W;

    m = lookup.dims.m;
    n = lookup.dims.n;
    p = lookup.dims.p;

    T = lookup.config.T;

    %% Generate a random control input
    PE_input = inputSignalGenerator(lookup, T);

    % Initialize input-output storage
    u_d = zeros(m, T);
    y_d = zeros(p, T);
    x_d = zeros(n, T + 1);
    y_d_w = zeros(p, T);
    x_d_w = zeros(n, T + 1);

    % Generate data by simulating the system on random inputs for L steps
    low = 1;
    if lookup.opt_params.init
        x_d(:, 1) = lookup.sys.params.x_ini;
        y_d(:, 1) = C * x_d(:, 1) + D * u_d(:, 1);
        u_d(:, 1) = PE_input(:, 1);
        low = 2;
    end
    
    T_d = lookup.T_d;
    for i = low:T
        if (i - T_d) < 1
            u_d(:, i) = 0;
        else
            u_d(:, i) = PE_input(:, i - T_d);
        end        
        w_i = W.c + W.G * (2*rand(size(W.G, 2), 1)-1);
        x_d(:, i + 1) = A * x_d(:, i) + B * u_d(:, i);
        x_d_w(:, i + 1) = x_d(:, i + 1) + w_i;
        y_d(:, i) = C * x_d(:, i) + D * u_d(:, i);
        y_d_w(:, i) = C * x_d_w(:, i) + D * u_d(:, i);
    end

    % Flatten the control inputs and outputs
    u = reshape(u_d, [], 1); % Reshapes into (T * sys.params.m) x 1
    y = reshape(y_d, [], 1); % Reshapes into (T * sys.params.p) x 1
    y_w = reshape(y_d, [], 1);
    x = reshape(x_d, [], 1);
    x_w = reshape(x_d_w, [], 1);

    nominal.x = x; nominal.y = y;
    perturbed.x = x_w; perturbed.y = y_w;
end

