function doublePlotDDSF(time, logs_nom, logs_pert, lookup)    
    sys = lookup.sys;
    ul_hist = logs_nom.ul_t; ul_hist_p = logs_pert.ul_t;
    u_hist = logs_nom.u; u_hist_p = logs_pert.u;
    y_hist = logs_nom.y; y_hist_p = logs_pert.y;
    yl_hist = logs_nom.yl; yl_hist_p = logs_pert.yl;
    output_dir = prepareOutputDir('plots');

    % --- Plot Learning Inputs vs Safe Inputs ---        
    figure(1);
    m = sys.dims.m;
    plots_per_figure = 3; % Max number of plots per figure
    num_figures = ceil(m / plots_per_figure); % Number of figures needed
    
    for fig = 1:num_figures
        figure('Position', [100, 100, 800, 600]); % Fixed figure size
                tiledlayout(plots_per_figure, 1); % Max 3 subplots per figure
    
            for sub = 1:plots_per_figure
                i = (fig - 1) * plots_per_figure + sub; % Global index for subplot
                if i > m, break; end % Stop if no more subplots are needed

                fprintf("time: %d x %d \n", size(time));
                fprintf("u_hist(%d, :): %d x % d\n", i, size(u_hist(i, :)));

                nexttile; grid on;
                stairs(time, ul_hist(i, :), 'r', 'LineStyle', ':','LineWidth', 1.75, 'DisplayName', sprintf('ul[%d]', i));
                hold on;
                stairs(time, u_hist(i, :), 'b', 'LineWidth', 1.25, 'DisplayName', sprintf('u[%d]', i));
            
                bounds = sys.constraints.U(i, :);
            
                % Plot boundaries
                if bounds(1) ~= -inf
                    plot(time, bounds(1) * ones(size(time)), 'b', 'LineStyle', '-.', 'DisplayName', 'Lower Bound');
                end
                if bounds(2) ~= inf
                    plot(time, bounds(2) * ones(size(time)), 'b', 'LineStyle', '--','DisplayName', 'Upper Bound');
                end
            
                title(sprintf('Learning vs Safe Input %d', i));
                xlabel('t');
                ylabel(sprintf('Input %d', i));
                grid on;
                legend show;
                hold off;
            end
            sgtitle('Learning Inputs vs. Safe Inputs');

             % Save the current figure
            prefix = sprintf('U-ddsf-%s-fig%d', lookup.systype, fig);
            saveas(gcf, fullfile(output_dir, strcat(prefix, '-plot.png')));
            matlab2tikz(fullfile(output_dir, strcat(prefix, '.tex')));
            close(gcf); 
    end        

   % --- Compare Nominal vs Perturbed Outputs ---
    figure('Position', [100, 100, 800, 600]);
    p = sys.dims.p;
    plots_per_figure = 3;
    num_figures = ceil(p / plots_per_figure);
    
    for fig = 1:num_figures
        figure('Position', [100, 100, 800, 600]);
        tiledlayout(plots_per_figure, 1);
    
        for sub = 1:plots_per_figure
            i = (fig - 1) * plots_per_figure + sub;
            if i > p, break; end
    
            nexttile; hold on; grid on;
    
            y_nom = logs_nom.y(i, :);
            y_pert = logs_pert.y(i, :);
    
            yl_nom = logs_nom.yl(i, :);
            yl_pert = logs_pert.yl(i, :);
    
            bounds = sys.constraints.Y(i, :);
    
            % Nominal
            stairs(time, y_nom, 'b', 'LineWidth', 1.5, 'DisplayName', 'y_{nom}');
            stairs(time, yl_nom, 'b--', 'LineWidth', 1.2, 'DisplayName', 'yl_{nom}');
    
            % Perturbed
            stairs(time, y_pert, 'r', 'LineWidth', 1.5, 'DisplayName', 'y_{pert}');
            stairs(time, yl_pert, 'r--', 'LineWidth', 1.2, 'DisplayName', 'yl_{pert}');
    
            % Constraints
            if bounds(1) ~= -inf
                plot(time, bounds(1) * ones(size(time)), 'k-.', 'DisplayName', 'Lower Bound');
            end
            if bounds(2) ~= inf
                plot(time, bounds(2) * ones(size(time)), 'k--', 'DisplayName', 'Upper Bound');
            end
    
            title(sprintf('Output %d: Nominal vs Perturbed', i));
            xlabel('Time'); ylabel(sprintf('y[%d]', i));
            legend('Location', 'best');
        end
        sgtitle(sprintf("Outputs Comparison (Fig %d)", fig));
    
        % Save
        prefix = sprintf('Y-nom-pert-%s-fig%d', lookup.systype, fig);
        output_dir = prepareOutputDir('plots');
        saveas(gcf, fullfile(output_dir, strcat(prefix, '-plot.png')));
        matlab2tikz(fullfile(output_dir, strcat(prefix, '.tex')));
        close(gcf);
    end
end

