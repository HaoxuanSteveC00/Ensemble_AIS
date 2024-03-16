clc;
% 2D Gaussian Mixtures
mu_list = [0 -3; 0 8; -4 4; 4 4];
sigma_list = [1.2, 0; 0, 0.01; 0.01, 0; 0, 2;
    0.2, 0; 0, 0.2; 0.2, 0; 0, 0.2];
weights = [0.25, 0.25, 0.25, 0.25];
sigma_list_gm = cat(3,[1.2 0.01],[0.01 2],[0.2 0.2],[0.2 0.2]);
gm = gmdistribution(mu_list, sigma_list_gm);
gmPDF = @(x,y) arrayfun(@(x0,y0) pdf(gm,[x0 y0]),x,y);
[X, Y] = meshgrid(-6:0.01:6, -6:0.01:12);
C = gmPDF(X,Y);

% Space and Time Parameters
dim = 2;
N_samples = 1000;
sample_set = linspace(1,N_samples,N_samples);
N_steps = 300;
t_grid = linspace(0,1,N_steps+1);
delta_t = 1.0 / N_steps;

% Initialization of Ensemble AIS
stretch = 10;
start_center = zeros(1, dim);
start_center_mat = repmat(start_center, N_samples, 1);
x_values = mvnrnd(start_center,eye(dim),N_samples);
loss_table = zeros(N_steps, 1);
mean_y_table = zeros(N_steps, 1);
sq_func_table = zeros(N_steps, 1);
x_values_c = x_values;
loss_table_c = zeros(N_steps, 1);
mean_y_table_c = zeros(N_steps, 1);
sq_func_table_c = zeros(N_steps, 1);

% Initialization of Standard AIS
inner_iter_AIS = 1;
x_values_AIS1 = x_values;
x_values_AIS2 = x_values;
log_weight_AIS1 = delta_t.*(0.5.*(power(vecnorm(x_values_AIS1-...
    start_center_mat,2,2), 2.0)) + log(pdf(gm, x_values_AIS1)));
log_weight_AIS2 = delta_t.*(0.5.*(power(vecnorm(x_values_AIS2-...
    start_center_mat,2,2), 2.0)) + log(pdf(gm, x_values_AIS2)));
loss_table_AIS1 = zeros(N_steps, 1);
mean_y_table_AIS1 = zeros(N_steps, 1);
sq_func_table_AIS1 = zeros(N_steps, 1);
loss_table_AIS2 = zeros(N_steps, 1);
mean_y_table_AIS2 = zeros(N_steps, 1);
sq_func_table_AIS2 = zeros(N_steps, 1);
true_mean_y = 3.25;
true_sq_func = (0.3+0.25*0.01+0.5*16.2)./3 +...
    (0.25*9.01+0.25*66+0.5*16.2)./5;

for i = 1:N_steps
    T = t_grid(i+1);
    fprintf('%d\n', T);

    % Ensemble AIS with exploration
    for j = 1:N_samples
        candid = sample_set(sample_set~=j);

        % MALA
        noise = mvnrnd(zeros(1, dim),eye(dim));
        drive = T.*GM_log_pdf(x_values(j,:), mu_list, sigma_list,...
            weights, gm) + (1.0 - T).*(x_values(j,:) - start_center);
        x_j_prop = x_values(j,:) -...
            delta_t.*drive + sqrt(2.*delta_t).*noise;

        drive_new = T.*GM_log_pdf(x_j_prop, mu_list, sigma_list,...
            weights, gm) + (1.0 - T).*(x_j_prop - start_center);
        prob_f = power(pdf(gm, x_j_prop), T).*power(exp(-power(...
            norm(x_j_prop - start_center), 2.0) / 2.0), 1.0 - T);
        move_f = exp(-power(norm(x_j_prop - x_values(j,:) +...
            delta_t.*drive), 2.0)/(4*delta_t));
        prob_s = power(pdf(gm, x_values(j,:)), T).*power(exp(-power(...
            norm(x_values(j,:) - start_center), 2.0) / 2.0), 1.0 - T);
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

        prob1 = power(pdf(gm, new_point), T);
        prob2 = power(pdf(gm, x_values(j,:)), T);
        prob3 = power(exp(-power(norm(new_point -...
            start_center), 2.0) / 2.0), 1.0 - T);
        prob4 = power(exp(-power(norm(x_values(j,:) -...
            start_center), 2.0) / 2.0), 1.0 - T);
        ratio = (z.^(dim-1)).*((prob1 * prob3) / (prob2 * prob4)); 
            
        if ratio > 1.0
            x_values(j,:) = new_point;
        elseif rand < ratio
            x_values(j,:) = new_point;
        end

        % Birth Death
        rate = -log(pdf(gm, x_values)) -...
            0.5.*(power(vecnorm(x_values-start_center_mat,2,2), 2.0));
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
        noise = mvnrnd(zeros(1, dim),eye(dim));
        drive = T.*GM_log_pdf(x_values_c(j,:), mu_list, sigma_list,...
            weights, gm)+(1.0 - T).*(x_values_c(j,:) - start_center);
        x_j_prop = x_values_c(j,:) - delta_t.*drive +...
            sqrt(2.*delta_t).*noise;

        drive_new = T.*GM_log_pdf(x_j_prop, mu_list, sigma_list,...
            weights, gm) + (1.0 - T).*(x_j_prop - start_center);
        prob_f = power(pdf(gm, x_j_prop), T).*power(exp(-power(...
            norm(x_j_prop - start_center), 2.0) / 2.0), 1.0 - T);
        move_f = exp(-power(norm(x_j_prop - x_values_c(j,:) +...
            delta_t.*drive), 2.0)/(4*delta_t));
        prob_s = power(pdf(gm, x_values_c(j,:)), T).*power(exp(-power(...
            norm(x_values_c(j,:) - start_center), 2.0) / 2.0), 1.0 - T);
        move_s = exp(-power(norm(x_values_c(j,:) - x_j_prop +...
            delta_t.*drive_new), 2.0)/(4*delta_t));
        ratio_Q = ((prob_f*move_s) / (prob_s*move_f));

        if ratio_Q > 1.0
            x_values_c(j,:) = x_j_prop;
        elseif rand < ratio_Q   
            x_values_c(j,:) = x_j_prop;
        end

        % Birth Death
        rate = -log(pdf(gm, x_values_c)) -...
            0.5.*(power(vecnorm(x_values_c-start_center_mat,2,2), 2.0));
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

    % Save empirical values
    loss_table(i) = -mean(log(pdf(gm, x_values)))-log(N_samples);
    loss_table_c(i) = -mean(log(pdf(gm, x_values_c)))-log(N_samples);
    mean_y_table(i) = mean(x_values(:,2));
    mean_y_table_c(i) = mean(x_values_c(:,2));
    sq_func_table(i) = mean((x_values(:,1).*x_values(:,1))./3 +...
        (x_values(:,2).*x_values(:,2))./5);
    sq_func_table_c(i) = mean((x_values_c(:,1).*x_values_c(:,1))./3 +...
        (x_values_c(:,2).*x_values_c(:,2))./5);

    % Standard AIS: MALA
    for j = 1:N_samples
        % MALA
        for k = 1:inner_iter_AIS
            noise = mvnrnd(zeros(1, dim),eye(dim));
            drive = T.*GM_log_pdf(x_values_AIS1(j,:), mu_list,...
                sigma_list,weights, gm)+(1.0 - T).*...
                (x_values_AIS1(j,:) - start_center);
            x_j_prop = x_values_AIS1(j,:) - delta_t.*drive + ...
                sqrt(2.*delta_t).*noise;
            
            drive_new = T.*GM_log_pdf(x_j_prop, mu_list, sigma_list,...
                weights, gm) + (1.0 - T).*(x_j_prop - start_center);
            prob_f = power(pdf(gm, x_j_prop), T).*power(exp(-power(...
                norm(x_j_prop - start_center), 2.0) / 2.0), 1.0 - T);
            move_f = exp(-power(norm(x_j_prop - x_values_AIS1(j,:) +...
                delta_t.*drive), 2.0)/(4*delta_t));
            prob_s = power(pdf(gm, x_values_AIS1(j,:)), T).*...
                power(exp(-power(norm(x_values_AIS1(j,:) -...
                start_center), 2.0) / 2.0), 1.0 - T);
            move_s = exp(-power(norm(x_values_AIS1(j,:) - x_j_prop +...
                delta_t.*drive_new), 2.0)/(4*delta_t));
            ratio_Q = ((prob_f*move_s) / (prob_s*move_f));

            if ratio_Q > 1.0
                x_values_AIS1(j,:) = x_j_prop;
            elseif rand < ratio_Q   
                x_values_AIS1(j,:) = x_j_prop;
            end
        end

        pt_new = x_values_AIS1(j,:);
        log_weight_AIS1(j) = log_weight_AIS1(j) +...
            delta_t.*(0.5.*power(norm(pt_new - start_center), 2.0) +...
            log(pdf(gm, pt_new)));
    end

    weight_AIS_list1 = exp(log_weight_AIS1)./sum(exp(log_weight_AIS1));
    loss_table_AIS1(i) = -sum(weight_AIS_list1.*...
        log(pdf(gm, x_values_AIS1))) +...
        sum(weight_AIS_list1.*log(weight_AIS_list1));
    mean_y_table_AIS1(i) = sum(weight_AIS_list1.*x_values_AIS1(:,2));
    sq_func_table_AIS1(i) = sum(weight_AIS_list1.*...
        ((x_values_AIS1(:,1).*x_values_AIS1(:,1))./3 +...
        (x_values_AIS1(:,2).*x_values_AIS1(:,2))./5));

    % Standard AIS: MH (Gaussian Kernel)
    for j = 1:N_samples
        pt = x_values_AIS2(j,:);
        for k = 1:inner_iter_AIS
            x_j_new = mvnrnd(pt, 0.01*eye(dim));
            prob_f = power(pdf(gm, x_j_new), T).*power(exp(-power(...
                norm(x_j_new - start_center), 2.0) / 2.0), 1.0 - T);
            prob_s = power(pdf(gm, pt), T).*power(exp(-power(norm(pt -...
                start_center), 2.0) / 2.0), 1.0 - T);
            ratio2 = prob_f / prob_s;

            if ratio2 > 1.0
                x_values_AIS2(j,:) = x_j_new;
            elseif rand < ratio2   
                x_values_AIS2(j,:) = x_j_new;
            end
        end

        pt_new = x_values_AIS2(j,:);
        log_weight_AIS2(j) = log_weight_AIS2(j) +...
            delta_t.*(0.5.*power(norm(pt_new - start_center), 2.0) +...
            log(pdf(gm, pt_new)));
    end

    weight_AIS_list2 = exp(log_weight_AIS2)./sum(exp(log_weight_AIS2));
    loss_table_AIS2(i) = -sum(weight_AIS_list2.*...
        log(pdf(gm, x_values_AIS2))) +...
        sum(weight_AIS_list2.*log(weight_AIS_list2));
    mean_y_table_AIS2(i) = sum(weight_AIS_list2.*x_values_AIS2(:,2));
    sq_func_table_AIS2(i) = sum(weight_AIS_list2.*...
        ((x_values_AIS2(:,1).*x_values_AIS2(:,1))./3 +...
        (x_values_AIS2(:,2).*x_values_AIS2(:,2))./5));
end    

figure(1);
contourf(X,Y,C);
xlabel('x');
ylabel('y');
colorbar;

figure(2);
scatter(x_values(:,1),x_values(:,2));
xlabel('x');
ylabel('y');
xlim([-6 6]);
ylim([-6 12]);

figure(3);
scatter(x_values_c(:,1),x_values_c(:,2), "red");
xlabel('x');
ylabel('y');
xlim([-6 6]);
ylim([-6 12]);

figure(4);
scatter(x_values_AIS1(:,1), x_values_AIS1(:,2), "green");
xlabel('x');
ylabel('y');
xlim([-6 6]);
ylim([-6 12]);

figure(5);
scatter(x_values_AIS2(:,1), x_values_AIS2(:,2), "black");
xlabel('x');
ylabel('y');
xlim([-6 6]);
ylim([-6 12]);

figure(6)
p61 = plot(loss_table);
hold on
p62 = plot(loss_table_c);
hold on
p63 = plot(loss_table_AIS1);
hold on
p64 = plot(loss_table_AIS2);
legend([p61 p62 p63 p64], 'MALA + Snooker + BD', 'MALA + BD',...
    'MALA + Reweight', 'Gaussian MH + Reweight', 'Location', 'Best');
xlabel('Time');
ylabel('Empirical KL Loss');

figure(7);
p71 = plot(mean_y_table);
hold on
p72 = plot(mean_y_table_c);
hold on
p73 = plot(mean_y_table_AIS1);
hold on
p74 = plot(mean_y_table_AIS2);
hold on
p75 = yline(true_mean_y,'--','Ground Truth');
legend([p71 p72 p73 p74], 'MALA + Snooker + BD', 'MALA + BD',...
    'MALA + Reweight', 'Gaussian MH + Reweight', 'Location', 'Best');
xlabel('Time');
ylabel('$\bf{E}[y]$','interpreter','latex');

figure(8);
p81 = plot(sq_func_table);
hold on
p82 = plot(sq_func_table_c);
hold on
p83 = plot(sq_func_table_AIS1);
hold on
p84 = plot(sq_func_table_AIS2);
hold on
p85 = yline(true_sq_func,'--','Ground Truth');
legend([p81 p82 p83 p84], 'MALA + Snooker + BD', 'MALA + BD',...
    'MALA + Reweight', 'Gaussian MH + Reweight', 'Location', 'Best');
xlabel('Time');
ylabel('$\bf{E}[\frac{1}{3}x^2 + \frac{1}{5}y^2]$','interpreter','latex');