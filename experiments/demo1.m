%% Step 1: Configuration
T_sim = 50; % Simulation length
data_options = struct( ...
    'datagen_mode', 'uniform', ...
    'scale', 1, ...
    'safe', false ...
);

run_options = struct( ...
    'system_type', 'rc_circuit', ...
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
    'discretize', false, ...
    'regularize', false, ...
    'constr_type', 'f', ...
    'solver_type', 'o', ...
    'target_penalty', false, ...
    'init', true, ...
    'R', 10 ...
);

%% Initialize the perturbed LTI system
sys = systemsPerturbedDTLTI(run_options.system_type);
dims = sys.dims;
opt_params.R = opt_params.R * eye(dims.m);

% Main lookup object
lookup = struct( ...
    'sys', sys, ...
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

y_d = reshape(nominal.y, dims.p, []);
u_d = reshape(u, dims.m, []);
[H_u, H_y] = hankelDDSF(u_d, y_d, lookup);

lookup.H = [H_u; H_y];
lookup.H_u = H_u;
lookup.H_y = H_y;
lookup.dims.hankel_cols = size(H_u, 2);

%% Step 3: Initialize Logs
T_ini = lookup.config.T_ini;
logs = struct( ...
    'u', [u_d(:, 1:(1+T_ini)).'; zeros(dims.m, T_sim).'].', ...
    'y', [y_d(:, 1:(1+T_ini)).'; zeros(dims.p, T_sim).'].', ...
    'ul', zeros(dims.m, lookup.config.N, T_sim), ...
    'ul_t', zeros(dims.m, T_sim), ...
    'loss', zeros(2, T_ini + T_sim), ...
    'y_noisy', zeros(dims.p, T_sim), ...
    'x_nom', zeros(dims.n, T_sim + 1), ...
    'x_noisy', zeros(dims.n, T_sim + 1) ...
);

logs.x_nom(:, 1) = sys.params.x_ini;
logs.x_noisy(:, 1) = sys.params.x_ini;

%% Step 4: Run Receding Horizon DDSF
for t = (T_ini + 1):(T_ini + T_sim)
    fprintf("===== TIME STEP %d =====\n", t - T_ini);
    
    u_l = learning_policy(lookup);
    logs.ul(:, :, t - T_ini) = u_l;
    ul_t = u_l(:, 1);
    logs.ul_t(:, t - T_ini) = ul_t;

    u_ini = logs.u(:, (t - T_ini):(t-1));
    y_ini = logs.y(:, (t - T_ini):(t-1));
    traj_ini = [u_ini; y_ini];

    [u_opt, y_opt] = optDDSF(lookup, u_l, traj_ini);
    loss_t = get_loss(lookup, ul_t, u_opt, y_opt);

    u_next = u_opt(:, 1 + T_ini);
    y_next = y_opt(:, 1 + T_ini);

    logs.u(:, t) = u_next;
    logs.y(:, t) = y_next;
    logs.loss(:, t) = loss_t;

    % Nominal + perturbed state update
    A = sys.A; B = sys.B; C = sys.C; D = sys.D;
    W = sys.params.W;

    x_prev_nom = logs.x_nom(:, t - T_ini);
    x_prev_noisy = logs.x_noisy(:, t - T_ini);
    w_k = W.c + W.G * (2 * rand(size(W.G, 2), 1) - 1);

    logs.x_nom(:, t - T_ini + 1) = A * x_prev_nom + B * u_next;
    logs.x_noisy(:, t - T_ini + 1) = A * x_prev_noisy + B * u_next + w_k;

    logs.y_noisy(:, t - T_ini) = C * logs.x_noisy(:, t - T_ini + 1) + D * u_next;

    % Update equilibrium sets
    lookup.sys.S_f.u_eq = [lookup.sys.S_f.u_eq, u_next];
    lookup.sys.S_f.y_eq = [lookup.sys.S_f.y_eq, y_next];
end

% Crop logs
logs.u = logs.u(:, T_ini + 1:end);
logs.y = logs.y(:, T_ini + 1:end);
lookup.logs = logs;

%% Step 5: Plotting Nominal vs Perturbed
time = 1:T_sim;
plotDDSF(time, logs, lookup); % optionally pass both y and y_noisy

% Add custom figure
figure;
tiledlayout(dims.p, 1);
for i = 1:dims.p
    nexttile;
    stairs(time, logs.y(i,:), 'b', 'LineWidth', 1.5); hold on;
    stairs(time, logs.y_noisy(i,:), 'r--', 'LineWidth', 1.5);
    legend('Nominal', 'Perturbed');
    xlabel('Time'); ylabel(sprintf('Output y[%d]', i));
    title(sprintf('Output Trajectory y[%d]: Nominal vs Perturbed', i));
    grid on;
end
