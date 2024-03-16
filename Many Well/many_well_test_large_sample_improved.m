clc;
% Parameters of many well distribution
dim = 20; 
beta = 0.001;
alpha = 0.1;

% Space and Time Parameters
N_samples = 3000; 
sample_set = linspace(1,N_samples,N_samples);
N_steps = 3000; 
t_grid = linspace(0,1,N_steps+1);
delta_t = 1.0 / N_steps;

% Initialization of Ensemble AIS
stretch = 10;
start_center = zeros(1,dim);
start_center_mat = repmat(start_center, N_samples, 1);
x_values = mvnrnd(start_center,eye(dim),N_samples);
loss_table = zeros(N_steps, 1);
x_values_c = x_values;
loss_table_c = loss_table;

x_BD_value = zeros(N_samples, 1);
x_BD_sum = 0;
for i = 1:N_samples
    fprintf('%d\n', i);
    BD_rate = -log(many_well_PDF(x_values(i,:), alpha, beta)) -...
            0.5.*(power(vecnorm(x_values(i,:)-start_center,2,2), 2.0));
    x_BD_value(i) = BD_rate;
    x_BD_sum = x_BD_sum + BD_rate;
end
x_BD_value_c = x_BD_value;
x_BD_sum_c = x_BD_sum;

% Initialization of Standard AIS
inner_iter_AIS = 1;
x_values_AIS1 = x_values;
x_values_AIS2 = x_values;
log_weight_AIS1 = delta_t.*(0.5.*(power(vecnorm(x_values_AIS1-...
    start_center_mat,2,2), 2.0)) +...
    log(many_well_PDF(x_values_AIS1, alpha, beta)));
log_weight_AIS2 = log_weight_AIS1;
loss_table_AIS1 = zeros(N_steps, 1);
loss_table_AIS2 = zeros(N_steps, 1);

for i = 1:N_steps
    T = t_grid(i+1);
    fprintf('%d\n', T);

    % Ensemble AIS with exploration
    for j = 1:N_samples
        candid = sample_set(sample_set~=j);

        % MALA
        noise = mvnrnd(zeros(dim,1),eye(dim));
        drive = T.*many_well_log_grad(x_values(j,:), alpha, beta) +...
            (1.0 - T).*(x_values(j,:) - start_center);
        x_j_prop = x_values(j,:) - delta_t.*drive +...
            sqrt(2.*delta_t).*noise;
        BD_rate_xj = x_BD_value(j);

        drive_new = T.*many_well_log_grad(x_j_prop, alpha, beta) +...
            (1.0 - T).*(x_j_prop - start_center);
        prob_f = power(many_well_PDF(x_j_prop, alpha, beta), T).*...
            power(exp(-power(norm(x_j_prop - start_center), 2.0)...
            / 2.0), 1.0 - T);
        move_f = exp(-power(norm(x_j_prop - x_values(j,:) +...
            delta_t.*drive), 2.0)/(4*delta_t));
        prob_s = power(many_well_PDF(x_values(j,:), alpha, beta), T).*...
            power(exp(-power(norm(x_values(j,:) - start_center), 2.0)...
            / 2.0), 1.0 - T);
        move_s = exp(-power(norm(x_values(j,:) - x_j_prop +...
            delta_t.*drive_new), 2.0)/(4*delta_t));
        ratio_Q = ((prob_f*move_s) / (prob_s*move_f));

        BD_rate_x_j_prop = -log(many_well_PDF(x_j_prop, alpha, beta)) -...
            0.5.*(power(vecnorm(x_j_prop-start_center,2,2), 2.0));

        if ratio_Q > 1.0
            x_values(j,:) = x_j_prop;
            % Update Birth Death Values
            x_BD_value(j) = BD_rate_x_j_prop;
            x_BD_sum = x_BD_sum + (BD_rate_x_j_prop - BD_rate_xj);
        elseif rand < ratio_Q   
            x_values(j,:) = x_j_prop;
            % Update Birth Death Values
            x_BD_value(j) = BD_rate_x_j_prop;
            x_BD_sum = x_BD_sum + (BD_rate_x_j_prop - BD_rate_xj);
        end

        % Snooker
        u = rand;
        z = power(u.*sqrt(stretch) + (1-u).*(sqrt(1./stretch)), 2.0);
        ind = randsample(candid, 1);
        new_point = (1-z).*x_values(ind,:) + z.*x_values(j,:);
        BD_rate_xj = x_BD_value(j);

        prob1 = power(many_well_PDF(new_point, alpha, beta), T);
        prob2 = power(many_well_PDF(x_values(j,:), alpha, beta), T);
        prob3 = power(exp(-power(norm(new_point - start_center),...
            2.0)/ 2.0), 1.0 - T);
        prob4 = power(exp(-power(norm(x_values(j,:) - start_center),...
            2.0) / 2.0), 1.0 - T);
        ratio = (z.^(dim-1)).*((prob1 * prob3) / (prob2 * prob4));
        BD_rate_new_point = -log(many_well_PDF(new_point, alpha, beta))...
            -0.5.*(power(vecnorm(new_point-start_center,2,2), 2.0)); 
            
        if ratio > 1.0
            x_values(j,:) = new_point;
            % Update Birth Death Values
            x_BD_value(j) = BD_rate_new_point;
            x_BD_sum = x_BD_sum + (BD_rate_new_point - BD_rate_xj);
        elseif rand < ratio
            x_values(j,:) = new_point;
            % Update Birth Death Values
            x_BD_value(j) = BD_rate_new_point;
            x_BD_sum = x_BD_sum + (BD_rate_new_point - BD_rate_xj);
        end

        % Birth Death
        BD_rate_xj = x_BD_value(j);
        rate_j = BD_rate_xj - (x_BD_sum./N_samples);
        pt = x_values(j,:);

        if rate_j > 0
            prob = 1 - exp(-rate_j * delta_t);
            if unifrnd(0,1) < prob
                rand_index = randsample(candid, 1);
                pt_rand = x_values(rand_index,:);
                x_values(j,:) = x_values(rand_index,:);
                BD_rate_rand = -log(many_well_PDF(pt_rand, alpha, beta))...
                    -0.5.*(power(vecnorm(pt_rand-start_center,2,2), 2.0));
                % Update Birth Death Values
                x_BD_value(j) = BD_rate_rand;
                x_BD_sum = x_BD_sum + (BD_rate_rand - BD_rate_xj);
            end
        elseif rate_j < 0
            prob = 1 - exp(rate_j * delta_t);
            if unifrnd(0,1) < prob
                rand_index = randsample(candid, 1);
                pt_rand = x_values(rand_index,:);
                BD_rate_rand = -log(many_well_PDF(pt_rand, alpha, beta))...
                    -0.5.*(power(vecnorm(pt_rand-start_center,2,2), 2.0));
                x_values(rand_index,:) = x_values(j,:);
                % Update Birth Death Values
                x_BD_value(rand_index) = BD_rate_xj;
                x_BD_sum = x_BD_sum + (BD_rate_xj - BD_rate_rand);
            end
        end    
    end

    % Ensemble AIS without exploration
    for j = 1:N_samples
        candid = sample_set(sample_set~=j);

        % MALA
        noise = mvnrnd(zeros(dim,1),eye(dim));
        drive = T.*many_well_log_grad(x_values_c(j,:), alpha, beta) +...
            (1.0 - T).*(x_values_c(j,:) - start_center);
        x_j_prop = x_values_c(j,:) - delta_t.*drive +...
            sqrt(2.*delta_t).*noise;
        BD_rate_xj_c = x_BD_value_c(j);

        drive_new = T.*many_well_log_grad(x_j_prop, alpha, beta) +...
            (1.0 - T).*(x_j_prop - start_center);
        prob_f = power(many_well_PDF(x_j_prop, alpha, beta), T).*...
            power(exp(-power(norm(x_j_prop - start_center), 2.0)...
            / 2.0), 1.0 - T);
        move_f = exp(-power(norm(x_j_prop - x_values_c(j,:) +...
            delta_t.*drive), 2.0)/(4*delta_t));
        prob_s = power(many_well_PDF(x_values_c(j,:), alpha, beta), T).*...
            power(exp(-power(norm(x_values_c(j,:) - start_center), 2.0)...
            / 2.0), 1.0 - T);
        move_s = exp(-power(norm(x_values_c(j,:) - x_j_prop +...
            delta_t.*drive_new), 2.0)/(4*delta_t));
        ratio_Q = ((prob_f*move_s) / (prob_s*move_f));

        BD_rate_x_j_prop_c = -log(many_well_PDF(x_j_prop, alpha, beta))...
            - 0.5.*(power(vecnorm(x_j_prop-start_center,2,2), 2.0));

        if ratio_Q > 1.0
            x_values_c(j,:) = x_j_prop;
            % Update Birth Death Values
            x_BD_value_c(j) = BD_rate_x_j_prop_c;
            x_BD_sum_c = x_BD_sum + (BD_rate_x_j_prop_c - BD_rate_xj_c);
        elseif rand < ratio_Q   
            x_values_c(j,:) = x_j_prop;
            % Update Birth Death Values
            x_BD_value_c(j) = BD_rate_x_j_prop_c;
            x_BD_sum_c = x_BD_sum + (BD_rate_x_j_prop_c - BD_rate_xj_c);
        end

        % Birth Death
        BD_rate_xj_c = x_BD_value_c(j);
        rate_j = BD_rate_xj_c - (x_BD_sum_c./N_samples);
        pt_c = x_values_c(j,:);

        if rate_j > 0
            prob = 1 - exp(-rate_j * delta_t);
            if unifrnd(0,1) < prob
                rand_index = randsample(candid, 1);
                pt_rand = x_values_c(rand_index,:);
                x_values_c(j,:) = x_values_c(rand_index,:);
                BD_rate_rand = -log(many_well_PDF(pt_rand, alpha, beta))...
                    -0.5.*(power(vecnorm(pt_rand-start_center,2,2), 2.0));
                % Update Birth Death Values
                x_BD_value_c(j) = BD_rate_rand;
                x_BD_sum_c = x_BD_sum_c + (BD_rate_rand - BD_rate_xj_c);
            end
        elseif rate_j < 0
            prob = 1 - exp(rate_j * delta_t);
            if unifrnd(0,1) < prob
                rand_index = randsample(candid, 1);
                pt_rand = x_values_c(rand_index,:);
                BD_rate_rand = -log(many_well_PDF(pt_rand, alpha, beta))...
                    -0.5.*(power(vecnorm(pt_rand-start_center,2,2), 2.0));
                x_values_c(rand_index,:) = x_values_c(j,:);
                % Update Birth Death Values
                x_BD_value_c(rand_index) = BD_rate_xj_c;
                x_BD_sum_c = x_BD_sum_c + (BD_rate_xj_c - BD_rate_rand);
            end
        end    
    end

    % Save values of empirical loss
    loss_table(i) = -mean(log(many_well_PDF(x_values, alpha, beta)))...
        -log(N_samples);
    loss_table_c(i) = -mean(log(many_well_PDF(x_values_c, alpha, beta)))...
        -log(N_samples);

    % Standard AIS: MALA
    for j = 1:N_samples
        for k = 1:inner_iter_AIS
            % MALA
            noise = mvnrnd(zeros(dim,1),eye(dim));
            drive = T.*many_well_log_grad(x_values_AIS1(j,:), alpha,...
                beta) + (1.0 - T).*(x_values_AIS1(j,:) - start_center);
            x_j_prop = x_values_AIS1(j,:) - delta_t.*drive +...
                sqrt(2.*delta_t).*noise;

            drive_new = T.*many_well_log_grad(x_j_prop, alpha,...
                beta) + (1.0 - T).*(x_j_prop - start_center);
            prob_f = power(many_well_PDF(x_j_prop, alpha, beta), T).*...
                power(exp(-power(norm(x_j_prop - start_center), 2.0)...
                / 2.0), 1.0 - T);
            move_f = exp(-power(norm(x_j_prop - x_values_AIS1(j,:) +...
                delta_t.*drive), 2.0)/(4*delta_t));
            prob_s = power(many_well_PDF(x_values_AIS1(j,:), alpha,...
                beta), T).*power(exp(-power(norm(x_values_AIS1(j,:) -...
                start_center), 2.0) / 2.0), 1.0 - T);
            move_s = exp(-power(norm(x_values_AIS1(j,:) -...
                x_j_prop + delta_t.*drive_new), 2.0)/(4*delta_t));
            ratio_Q = ((prob_f*move_s) / (prob_s*move_f));

            if ratio_Q > 1.0
                x_values_AIS1(j,:) = x_j_prop;
            elseif rand < ratio_Q   
                x_values_AIS1(j,:) = x_j_prop;
            end

            pt_new = x_values_AIS1(j,:);
            log_weight_AIS1(j) = log_weight_AIS1(j) +...
                delta_t.*(0.5.*(power(norm(pt_new - start_center),...
                2.0)) + log(many_well_PDF(pt_new, alpha, beta)));
        end
    end

    % Save values of empirical loss
    weight_AIS_list1 = exp(log_weight_AIS1)./sum(exp(log_weight_AIS1));
    loss_table_AIS1(i) = -sum(weight_AIS_list1.*...
        log(many_well_PDF(x_values_AIS1, alpha, beta))) +...
        sum(weight_AIS_list1.*log(weight_AIS_list1));

    % Standard AIS: MH (Gaussian Kernel)
    for j = 1:N_samples
        pt = x_values_AIS2(j,:);
        for k = 1:inner_iter_AIS
            x_j_new = mvnrnd(pt, 0.01*eye(dim));
            prob_f = power(many_well_PDF(x_j_new, alpha, beta), T).*...
                power(exp(-power(norm(x_j_new - start_center), 2.0)...
                / (2.0)), 1.0 - T);
            prob_s = power(many_well_PDF(pt, alpha, beta), T).*...
                power(exp(-power(norm(pt - start_center), 2.0)...
                /(2.0)), 1.0 - T);
            ratio2 = prob_f / prob_s;

            if ratio2 > 1.0
                x_values_AIS2(j,:) = x_j_new;
            elseif rand < ratio2   
                x_values_AIS2(j,:) = x_j_new;
            end
        end

        pt_new = x_values_AIS2(j,:);
        log_weight_AIS2(j) = log_weight_AIS2(j) +...
            delta_t.*(0.5.*(power(norm(pt_new - start_center),...
            2.0)) + log(many_well_PDF(pt_new, alpha, beta)));
    end

    % Save values of empirical loss
    weight_AIS_list2 = exp(log_weight_AIS2)./sum(exp(log_weight_AIS2));
    loss_table_AIS2(i) = -sum(weight_AIS_list2.*...
        log(many_well_PDF(x_values_AIS2, alpha, beta))) +...
        sum(weight_AIS_list2.*log(weight_AIS_list2));
end    

figure(1);
p11 = plot(loss_table);
hold on
p12 = plot(loss_table_c);
hold on
p13 = plot(loss_table_AIS1);
hold on
p14 = plot(loss_table_AIS2);
legend([p11 p12 p13 p14], 'MALA + Snooker + BD', 'MALA + BD',...
    'MALA + Reweight', 'Gaussian MH + Reweight', 'Location', 'Best');
xlabel('Time');
ylabel('Empirical KL Loss');

figure(2);
scatter(x_values(:,1),x_values(:,2));
xlabel('$x_1$','interpreter','latex');
ylabel('$x_2$','interpreter','latex');
xlim([-10 10]);
ylim([-10 10]);

figure(3);
scatter(x_values_c(:,1),x_values_c(:,2), "red");
xlabel('$x_1$','interpreter','latex');
ylabel('$x_2$','interpreter','latex');
xlim([-10 10]);
ylim([-10 10]);

figure(4);
scatter(x_values_AIS1(:,1), x_values_AIS1(:,2), "green");
xlabel('$x_1$','interpreter','latex');
ylabel('$x_2$','interpreter','latex');
xlim([-10 10]);
ylim([-10 10]);

figure(5);
scatter(x_values_AIS2(:,1), x_values_AIS2(:,2), "black");
xlabel('$x_1$','interpreter','latex');
ylabel('$x_2$','interpreter','latex');
xlim([-10 10]);
ylim([-10 10]);