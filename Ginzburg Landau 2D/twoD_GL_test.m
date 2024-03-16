clc;
% Parameters of Ginzburg Landau distribution
dim = 16;
dim0 = sqrt(dim);
delta = 0.125;
beta = 10;
h = 1.0./(dim0+1);

% Space and Time Parameters
N_samples = 1000; 
sample_set = linspace(1,N_samples,N_samples);
N_steps = 150;
t_grid = linspace(0,1,N_steps+1);
delta_t = 1.0 / N_steps;

% Initialization of Ensemble AIS
stretch = 10;
start_center = zeros(1, dim);
sigmas_init = 0.01; %1.0
start_center_mat = repmat(start_center, N_samples, 1);
x_values = mvnrnd(start_center,eye(dim).*sigmas_init,N_samples);
loss_table = zeros(N_steps, 1);
x_values_c = x_values;
loss_table_c = zeros(N_steps, 1);

% Initialization of Standard AIS
inner_iter_AIS = 1;
x_values_AIS1 = x_values;
x_values_AIS2 = x_values;
log_weight_AIS1 = delta_t.*(0.5.*(power(vecnorm(x_values_AIS1-...
    start_center_mat,2,2), 2.0)./sigmas_init) +...
    log(twoD_GL_PDF(x_values_AIS1,delta, beta, h)));
log_weight_AIS2 = delta_t.*(0.5.*(power(vecnorm(x_values_AIS2-...
    start_center_mat,2,2), 2.0)./sigmas_init) +...
    log(twoD_GL_PDF(x_values_AIS2, delta, beta, h)));
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
        drive = -T.*twoD_GL_grad(x_values(j,:), delta, beta, h) +...
            (1.0 - T).*((x_values(j,:) - start_center)./sigmas_init);
        x_j_prop = x_values(j,:) - delta_t.*drive +...
            sqrt(2.*delta_t).*noise;

        drive_new = -T.*twoD_GL_grad(x_j_prop, delta, beta, h) +...
            (1.0 - T).*((x_j_prop - start_center)./(sigmas_init));
        prob_f = power(twoD_GL_PDF(x_j_prop, delta, beta, h), T).*...
            power(exp(-power(norm(x_j_prop - start_center), 2.0) /...
            (2.0*sigmas_init)), 1.0 - T);
        move_f = exp(-power(norm(x_j_prop - x_values(j,:) +...
            delta_t.*drive), 2.0)/(4*delta_t));
        prob_s = power(twoD_GL_PDF(x_values(j,:), delta, beta, h),...
            T).*power(exp(-power(norm(x_values(j,:) - start_center),...
            2.0) /(2.0*sigmas_init)), 1.0 - T);
        move_s = exp(-power(norm(x_values(j,:) - x_j_prop +...
            delta_t.*drive_new), 2.0)/(4*delta_t));
        ratio_Q = ((prob_f*move_s) / (prob_s*move_f));

        if ratio_Q > 1.0
            x_values(j,:) = x_j_prop;
        elseif rand < ratio_Q   
            x_values(j,:) = x_j_prop;
        end

        % Snooker
        u = rand;
        z = power(u.*sqrt(stretch) + (1-u).*(sqrt(1./stretch)), 2.0);
        ind = randsample(candid, 1);
        new_point = (1-z).*x_values(ind,:) + z.*x_values(j,:);

        prob1 = power(twoD_GL_PDF(new_point, delta, beta, h), T);
        prob2 = power(twoD_GL_PDF(x_values(j,:), delta, beta, h), T);
        prob3 = power(exp(-power(norm(new_point -...
            start_center), 2.0) / (2.0*sigmas_init)), 1.0 - T);
        prob4 = power(exp(-power(norm(x_values(j,:) -...
            start_center), 2.0) / (2.0*sigmas_init)), 1.0 - T);
        ratio = (z.^(dim-1)).*((prob1 * prob3) / (prob2 * prob4)); 
            
        if ratio > 1.0
            x_values(j,:) = new_point;
        elseif rand < ratio
            x_values(j,:) = new_point;
        end

        % Birth Death
        rate = -log(twoD_GL_PDF(x_values, delta, beta, h)) -...
            0.5.*(power(vecnorm(x_values-start_center_mat,2,2),...
            2.0)./(sigmas_init));
        rate_j = rate(j) - mean(rate);

        if rate_j > 0
            prob = 1 - exp(-rate_j * delta_t);
            if unifrnd(0,1) < prob
                rand_index = randsample(candid, 1);
                x_values(j,:) = x_values(rand_index,:);
            end
        elseif rate_j < 0
            prob = 1 - exp(rate_j * delta_t);
            if unifrnd(0,1) < prob
                rand_index = randsample(candid, 1);
                x_values(rand_index,:) = x_values(j,:);
            end
        end    
    end

    % Ensemble AIS without exploration
    for j = 1:N_samples
        candid = sample_set(sample_set~=j);

        % MALA
        noise = mvnrnd(zeros(dim,1),eye(dim));
        drive = -T.*twoD_GL_grad(x_values_c(j,:), delta, beta, h) +...
            (1.0 - T).*((x_values_c(j,:) - start_center)./sigmas_init);
        x_j_prop = x_values_c(j,:) - delta_t.*drive +...
            sqrt(2.*delta_t).*noise;

        drive_new = -T.*twoD_GL_grad(x_j_prop, delta, beta, h) +...
            (1.0 - T).*((x_j_prop - start_center)./sigmas_init);
        prob_f = power(twoD_GL_PDF(x_j_prop, delta, beta, h), T).*...
            power(exp(-power(norm(x_j_prop - start_center), 2.0) /...
            (2.0*sigmas_init)), 1.0 - T);
        move_f = exp(-power(norm(x_j_prop - x_values_c(j,:) +...
            delta_t.*drive), 2.0)/(4*delta_t));
        prob_s = power(twoD_GL_PDF(x_values_c(j,:), delta, beta, h),...
            T).*power(exp(-power(norm(x_values_c(j,:) -...
            start_center),2.0) / (2.0*sigmas_init)), 1.0 - T);
        move_s = exp(-power(norm(x_values_c(j,:) - x_j_prop +...
            delta_t.*drive_new), 2.0)/(4*delta_t));
        ratio_Q = ((prob_f*move_s) / (prob_s*move_f));

        if ratio_Q > 1.0
            x_values_c(j,:) = x_j_prop;
        elseif rand < ratio_Q   
            x_values_c(j,:) = x_j_prop;
        end

        % Birth Death
        rate = -log(twoD_GL_PDF(x_values_c, delta, beta, h)) -...
            0.5.*(power(vecnorm(x_values_c-start_center_mat,2,2),...
            2.0)./(sigmas_init));
        rate_j = rate(j) - mean(rate);

        if rate_j > 0
            prob = 1 - exp(-rate_j * delta_t);
            if unifrnd(0,1) < prob
                rand_index = randsample(candid, 1);
                x_values_c(j,:) = x_values_c(rand_index,:);
            end
        elseif rate_j < 0
            prob = 1 - exp(rate_j * delta_t);
            if unifrnd(0,1) < prob
                rand_index = randsample(candid, 1);
                x_values_c(rand_index,:) = x_values_c(j,:);
            end
        end    
    end

    loss_table(i) = -mean(log(twoD_GL_PDF(x_values, delta, beta, h)))...
        - log(N_samples);
    loss_table_c(i) = -mean(log(twoD_GL_PDF(x_values_c, delta, beta,...
        h))) - log(N_samples);

    % Standard AIS: MALA
    for j = 1:N_samples
        for k = 1:inner_iter_AIS
            noise = mvnrnd(zeros(dim,1),eye(dim));
            drive = -T.*twoD_GL_grad(x_values_AIS1(j,:), delta, beta,...
                h) + (1.0 - T).*((x_values_AIS1(j,:) -...
                start_center)./sigmas_init);
            x_j_prop = x_values_AIS1(j,:) - delta_t.*drive +...
                sqrt(2.*delta_t).*noise;

            drive_new = -T.*twoD_GL_grad(x_j_prop, delta, beta, h) +...
                (1.0 - T).*((x_j_prop - start_center)./(sigmas_init));
            prob_f = power(twoD_GL_PDF(x_j_prop, delta, beta, h),...
                T).*power(exp(-power(norm(x_j_prop - start_center),...
                2.0)/ (2.0*sigmas_init)), 1.0 - T);
            move_f = exp(-power(norm(x_j_prop - x_values_AIS1(j,:) +...
                delta_t.*drive), 2.0)/(4*delta_t));
            prob_s = power(twoD_GL_PDF(x_values_AIS1(j,:), delta,...
                beta, h), T).*power(exp(-power(norm(...
                x_values_AIS1(j,:) - start_center), 2.0) /...
                (2.0*sigmas_init)), 1.0 - T);
            move_s = exp(-power(norm(x_values_AIS1(j,:) - x_j_prop +...
                delta_t.*drive_new), 2.0)/(4*delta_t));
            ratio_Q = ((prob_f*move_s) / (prob_s*move_f));

            if ratio_Q > 1.0
                x_values_AIS1(j,:) = x_j_prop;
            elseif rand < ratio_Q   
                x_values_AIS1(j,:) = x_j_prop;
            end

            pt_new = x_values_AIS1(j,:);
            log_weight_AIS1(j) = log_weight_AIS1(j) +...
                delta_t.*(0.5.*(power(norm(pt_new - start_center),...
                2.0))./sigmas_init + log(twoD_GL_PDF(pt_new,...
                delta, beta, h)));
        end    
    end    

    weight_AIS_list1 = exp(log_weight_AIS1)./sum(exp(log_weight_AIS1));
    loss_table_AIS1(i) = -sum(weight_AIS_list1.*...
        log(twoD_GL_PDF(x_values_AIS1, delta, beta, h))) +...
        sum(weight_AIS_list1.*log(weight_AIS_list1));

    % Standard AIS: MH (Gaussian Kernel)
    for j = 1:N_samples
        pt = x_values_AIS2(j,:);
        for k = 1:inner_iter_AIS
            x_j_new = mvnrnd(pt, 0.001*eye(dim));
            prob_f = power(twoD_GL_PDF(x_j_new, delta, beta, h), T).*...
                power(exp(-power(norm(x_j_new - start_center), 2.0)...
                / (2.0*sigmas_init)), 1.0 - T);
            prob_s = power(twoD_GL_PDF(pt, delta, beta, h), T).*...
                power(exp(-power(norm(pt - start_center), 2.0)...
                /(2.0*sigmas_init)), 1.0 - T);
            ratio2 = prob_f / prob_s;

            if ratio2 > 1.0
                x_values_AIS2(j,:) = x_j_new;
            elseif rand < ratio_Q   
                x_values_AIS2(j,:) = x_j_new;
            end
        end

        pt_new = x_values_AIS2(j,:);
        log_weight_AIS2(j) = log_weight_AIS2(j) +...
                delta_t.*(0.5.*(power(norm(pt_new - start_center),...
                2.0))./sigmas_init + log(twoD_GL_PDF(pt_new,...
                delta, beta, h)));
    end

    weight_AIS_list2 = exp(log_weight_AIS2)./sum(exp(log_weight_AIS2));
    loss_table_AIS2(i) = -sum(weight_AIS_list2.*...
        log(twoD_GL_PDF(x_values_AIS2, delta, beta, h))) +...
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
scatter(x_values(:,5),x_values(:,6));
xlabel('$x_5$','interpreter','latex');
ylabel('$x_6$','interpreter','latex');
xlim([-3 3]);
ylim([-3 3]);

figure(3);
scatter(x_values_c(:,5),x_values_c(:,6), "red");
xlabel('$x_5$','interpreter','latex');
ylabel('$x_6$','interpreter','latex');
xlim([-3 3]);
ylim([-3 3]);

figure(4);
scatter(x_values_AIS1(:,5), x_values_AIS1(:,6), "green");
xlabel('$x_5$','interpreter','latex');
ylabel('$x_6$','interpreter','latex');
xlim([-3 3]);
ylim([-3 3]);

figure(5);
scatter(x_values_AIS2(:,5), x_values_AIS2(:,6), "black");
xlabel('$x_5$','interpreter','latex');
ylabel('$x_6$','interpreter','latex');
xlim([-3 3]);
ylim([-3 3]);