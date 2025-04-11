%% Step 1: Configuration
systype = 'quadrotor';
T_sim = 10; % Simulation length
data_options = struct( ...
    'datagen_mode', 'controlled_random', ...
    'scale', 1, ...
    'safe', false ...
);

run_options = struct( ...
    'system_type', systype, ...
    'T_sim', T_sim, ...
    'T_d', 0 ...
);

IO_params = struct( ...
    'debug', true, ...
    'save', true, ...
    'log_interval', 1, ...
    'verbose', false ...
);

opt_params = struct( ...
    'discretize', true, ...
    'regularize', true, ...
    'constr_type', 'f', ...
    'solver_type', 'o', ...
    'target_penalty', false, ...
    'init', true, ...
    'R', 1 ...
);

%% Initialize the perturbed LTI system
sys = systemsPerturbedDTLTI(run_options.system_type);
dims = sys.dims;
opt_params.R = opt_params.R * eye(dims.m);

% Main lookup object
lookup = struct( ...
    'sys', sys, ...
    'systype', systype, ...
    'opt_params', opt_params, ...
    'config', sys.config, ...
    'dims', dims, ...
    'IO_params', IO_params, ...
    'T_sim', run_options.T_sim, ...
    'data_options', data_options, ...
    'T_d', run_options.T_d ...
);

%% Step 2: Data Generation and Hankel
[perturbed, nominal, u] = noisyDataGen(lookup);

y_d_nominal = reshape(nominal.y, dims.p, []);
y_d_perturbed = reshape(perturbed.y, dims.p, []);
u_d = reshape(u, dims.m, []);
[H_u, H_y] = hankelDDSF(u_d, y_d_nominal, lookup);
[H_u_perturbed, H_y_perturbed] = hankelDDSF(u_d, y_d_perturbed, lookup);

lookup.H = [H_u; H_y];
lookup.H_perturbed = [H_u_perturbed; H_y_perturbed];

lookup.H_u = H_u;
lookup.H_y = H_y;

lookup.H_u_perturbed = H_u_perturbed;
lookup.H_y_p = H_y_perturbed;

lookup.dims.hankel_cols = size(H_u, 2);

H_u_2 = construct_hankel(u_d, 2);
H_y_2 = construct_hankel(y_d_nominal, 2);
H_y_2_p = construct_hankel(y_d_perturbed, 2);

%% Step 3: Initialize Logs
T_ini = lookup.config.T_ini;
logs = struct( ...
        'u', [u_d(:, 1:(1+ T_ini)).'; ...
        zeros(dims.m, T_sim).'].', ... 
        'y', [y_d_nominal(:, 1:(1 + T_ini)).'; ...
        zeros(dims.p, T_sim).'].', ... 
        'yl', zeros(dims.p, T_sim), ...
        'x', zeros(dims.n, T_ini + T_sim), ...
        'ul', zeros(dims.m, lookup.config.N, T_sim), ...
        'ul_t', zeros(dims.m, T_sim), ...
        'loss', zeros(2, T_ini + T_sim) ...
    );

logs_perturbed = struct( ...
        'u', [u_d(:, 1:(1+ T_ini)).'; ...
        zeros(dims.m, T_sim).'].', ... 
        'y', [y_d_perturbed(:, 1:(1 + T_ini)).'; ...
        zeros(dims.p, T_sim).'].', ... 
        'yl', zeros(dims.p, T_sim), ...
        'x', zeros(dims.n, T_ini + T_sim), ...
        'ul', zeros(dims.m, lookup.config.N, T_sim), ...
        'ul_t', zeros(dims.m, T_sim), ...
        'loss', zeros(2, T_ini + T_sim) ...
    );

%logs.x(:, 1:T_ini) = sys.params.x_ini;
%logs_perturbed.x(:, 1:T_ini) = sys.params.x_ini;

T_d = run_options.T_d;
%% Step 4: Run Receding Horizon DDSF
for t = (T_ini + 1):(T_ini + T_sim)
    fprintf("===== TIME STEP %d / %d =====\n", t - T_ini, T_sim);
    
    u_l = learning_policy(lookup);
    logs.ul(:, :, t - T_ini) = u_l; logs_perturbed.ul(:, :, t - T_ini) = u_l;
    ul_t = u_l(:, 1);
    logs.ul_t(:, t - T_ini) = ul_t; logs_perturbed.ul_t(:, t - T_ini) = ul_t;

    u_ini = logs.u(:, (t - T_ini):(t-1));
    y_ini = logs.y(:, (t - T_ini):(t-1)); y_ini_p = logs_perturbed.y(:, (t - T_ini):(t-1));
    traj_ini = [u_ini; y_ini]; traj_ini_p = [u_ini; y_ini_p];

    [u_opt, y_opt] = optDDSF(lookup, u_l, traj_ini);
    loss_t = get_loss(lookup, ul_t, u_opt, y_opt);

    [u_opt_p, y_opt_p] = optDDSF(lookup, u_l, traj_ini_p);
    loss_t_p = get_loss(lookup, ul_t, u_opt_p, y_opt_p);

    u_next = u_opt(:, 1 + T_ini); u_next_p = u_opt_p(:, 1 + T_ini);
    y_next = y_opt(:, 1 + T_ini); y_next_p = y_opt_p(:, 1 + T_ini);

    logs.u(:, t) = u_next; logs_perturbed.u(:, t) = u_next_p;
    logs.y(:, t) = y_next; logs_perturbed.y(:, t) = y_next_p;
    logs.loss(:, t) = loss_t; logs_perturbed.loss(:, t) = loss_t_p;

    % Nominal + perturbed state update
    A = sys.A; B = sys.B; C = sys.C; D = sys.D;
    W = sys.params.W;

    x_prev_nom = logs.x(:, t - T_ini);
    x_prev_noisy = logs_perturbed.x(:, t - T_ini);
    w_k = W.c + W.G * (2 * rand(size(W.G, 2), 1) - 1);

    logs.x(:, t - T_ini + 1) = A * x_prev_nom + B * u_next;
    logs_perturbed.x(:, t - T_ini + 1) = A * x_prev_noisy + B * u_next + w_k;
    
    try            
        yl_next = dataBasedU2Y(ul_t, logs.u(:, t),logs.y(:, t), H_u_2, H_y_2);
        logs.yl(:, t-T_ini) = yl_next;

        yl_next_p = dataBasedU2Y(ul_t, logs.u(:, t),logs_perturbed.y(:, t), H_u_2, H_y_2_p);
        logs_perturbed.yl(:, t-T_ini) = yl_next_p;
    catch ME
        fprintf('Failed to execute dataBasedU2Y at time step %d: %s', t-T_ini, ME.message);
    end

    % Update equilibrium sets
    %lookup.sys.S_f.u_eq = [lookup.sys.S_f.u_eq, u_next];
    %lookup.sys.S_f.y_eq = [lookup.sys.S_f.y_eq, y_next];
end

% Crop logs
logs.u = logs.u(:, T_ini + 1:end-1);
logs.y = logs.y(:, T_ini + 1:end-1);
logs_perturbed.u = logs_perturbed.u(:, T_ini + 1:end-1);
logs_perturbed.y = logs_perturbed.y(:, T_ini + 1:end-1);
lookup.logs = logs;
lookup.logs_perturbed = logs_perturbed;

%% Step 5: Plotting Nominal vs Perturbed
time = 1:T_sim;
doublePlotDDSF(time, logs, logs_perturbed, lookup); 

function loss = get_loss(lookup, u_l, u_opt, y_opt)
    R = lookup.opt_params.R;
    y_target = lookup.sys.params.target;

    loss1 = det(R) * norm(u_l - u_opt);    
    loss2 = loss1 + norm(y_target - y_opt);
    
    loss = [loss1; loss2];
end