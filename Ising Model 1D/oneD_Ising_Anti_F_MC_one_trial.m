dim = 20;
num_supp = 2^dim;
basis = 2.^(((dim-1):-1:0));
Mat_f = -diag(ones(dim-1,1)./2, 1) -diag(ones(dim-1,1)./2, -1)-...
    diag(ones(dim-2,1)./6, 2)-diag(ones(dim-2,1)./6, -2); 
Mat_f = -Mat_f;

N_sample_list = 2.^([6,7,8,9,10]);
error_list_1 = zeros(1,5);
error_list_2 = zeros(1,5);
error_list_3 = zeros(1,5);
error_list_6 = zeros(1,5);

N_steps = 64;
t_grid = linspace(0,1,N_steps+1);
delta_t = 1.0 / N_steps;
inner_iter = 1;
inner_iter_AIS = 3;
beta = 0.8;

E_f_list = zeros(num_supp, 1);
for l = 0:(num_supp-1)
    vec_l = dec2bin(l,dim)-'0';
    vec_l = 2*vec_l-1;
    E_f_list(l+1) = exp(-beta.*(vec_l*Mat_f*(vec_l')));
end
prob_f_list = E_f_list./(sum(E_f_list));
ref_prob = ones(num_supp,1);
ref_prob = ref_prob./sum(ref_prob);

for ind_s = 1:5
    N_samples = N_sample_list(ind_s);
    sample_set = linspace(1,N_samples,N_samples);
    emp_data = zeros(N_samples, dim);
    for i = 1:N_samples
        ind = randsample(1:1:num_supp,1,true,ref_prob) - 1;
        ind_vec = dec2bin(ind, dim) - '0';
        ind_vec = 2.*ind_vec -1;
        emp_data(i,:) = ind_vec;
    end    
    emp_data_c = emp_data;
    emp_data_AIS = emp_data;
    log_weight_list = delta_t.*ones(N_samples, 1) -...
        delta_t*beta*(diag(emp_data_AIS*Mat_f*(emp_data_AIS')));

    for i = 1:N_steps
        T = t_grid(i+1);
        fprintf('%d\n', T);

        % Ensemble AIS with exploration
        for k = 1:inner_iter
            for j = 1:N_samples
                candid = sample_set(sample_set~=j);
                pt = emp_data(j,:);

                % Glauber Dynamics
                comp = unidrnd(dim);
                pt_flip = pt;
                pt_flip(comp) = -pt(comp);

                r1 = exp(-beta.*T.*(pt*Mat_f*(pt'))); 
                r2 = exp(-beta.*T.*(pt_flip*Mat_f*(pt_flip'))); 
                ratio = r2./(r1+r2);
                if rand < ratio
                    emp_data(j,:) = pt_flip;
                end    

                % Genetic Algorithm
                rand_ind = randsample(candid, 1);
                pt = emp_data(j,:);
                pt_r = emp_data(rand_ind,:);
                r_pt = exp(-beta.*T.*(pt*Mat_f*(pt'))); 
                r_pt_r = exp(-beta.*T.*(pt_r*Mat_f*(pt_r'))); 

                data_y1 = zeros(1,dim);
                data_y2 = zeros(1,dim);
                for l = 1:dim
                    if rand < 0.5
                        data_y1(l) = pt(l);
                        data_y2(l) = pt_r(l);
                    else
                        data_y1(l) = pt_r(l);
                        data_y2(l) = pt(l);
                    end    
                end
                r_y1 = exp(-beta.*T.*(data_y1*Mat_f*(data_y1'))); 
                r_y2 = exp(-beta.*T.*(data_y2*Mat_f*(data_y2'))); 
                rate = min(1,(r_y1*r_y2)./(r_pt*r_pt_r));

                if rand < rate
                    emp_data(j,:) = data_y1;
                    emp_data(rand_ind,:) = data_y2;
                end    

                % Birth Death
                rate = beta.*diag(emp_data*Mat_f*emp_data'); 
                rate_j = rate(j) - mean(rate);
                pt = emp_data(j,:);

                if rate_j > 0
                    prob = 1 - exp(-rate_j * delta_t);
                    if rand < prob
                        rand_index = randsample(candid, 1);
                        pt_s = emp_data(rand_index,:);
                        emp_data(j,:) = pt_s;
                    end
                elseif rate_j < 0
                    prob = 1 - exp(rate_j * delta_t);
                    if rand < prob
                        rand_index = randsample(candid, 1);
                        emp_data(rand_index,:) = pt;
                    end
                end

            end
        end

        % Ensemble AIS without exploration
        for k = 1:inner_iter
            for j = 1:N_samples
                candid = sample_set(sample_set~=j);
                pt = emp_data_c(j,:);

                % Glauber Dynamics
                comp = unidrnd(dim);
                pt_flip = pt;
                pt_flip(comp) = -pt(comp);
                r1 = exp(-beta.*T.*(pt*Mat_f*(pt'))); 
                r2 = exp(-beta.*T.*(pt_flip*Mat_f*(pt_flip'))); 
                ratio = r2./(r1+r2);
                if rand < ratio
                    emp_data_c(j,:) = pt_flip;
                end

                % Birth Death
                rate = beta.*diag(emp_data_c*Mat_f*emp_data_c'); 
                rate_j = rate(j) - mean(rate);
                pt = emp_data_c(j,:);

                if rate_j > 0
                    prob = 1 - exp(-rate_j * delta_t);
                    if rand < prob
                        rand_index = randsample(candid, 1);
                        pt_s = emp_data_c(rand_index,:);
                        emp_data_c(j,:) = pt_s;
                    end
                elseif rate_j < 0
                    prob = 1 - exp(rate_j * delta_t);
                    if rand < prob
                        rand_index = randsample(candid, 1);
                        emp_data_c(rand_index,:) = pt;
                    end
                end

            end
        end

        % Standard AIS with Glauber dynamics
        for j = 1:N_samples
            pt = emp_data_AIS(j,:);
            for k = 1:inner_iter_AIS
                pt = emp_data_AIS(j,:);
                % Glauber Dynamics
                comp = unidrnd(dim);
                pt_flip = pt;
                pt_flip(comp) = -pt(comp);
                r1 = exp(-beta.*T.*(pt*Mat_f*(pt'))); 
                r2 = exp(-beta.*T.*(pt_flip*Mat_f*(pt_flip'))); 
                ratio = r2./(r1+r2);
                if rand < ratio
                    emp_data_AIS(j,:) = pt_flip;
                end
            end

            pt_new = emp_data_AIS(j,:);
            % log weight update
            log_weight_list(j) = log_weight_list(j) -...
                delta_t*beta*(pt_new*Mat_f*(pt_new')) + delta_t;
        end
    end  

    p_proba = zeros(num_supp, 1);
    for i = 1:N_samples
        ind = randsample(1:1:num_supp,1,true,prob_f_list);
        p_proba(ind) = p_proba(ind) + 1;
    end
    p_proba_ren = p_proba./sum(p_proba);
    error_list_6(ind_s) = norm(prob_f_list - p_proba_ren);

    weight_AIS_list = exp(log_weight_list)./sum(exp(log_weight_list));
    p_AIS_proba = zeros(num_supp, 1);
    for i = 1:N_samples
        v_ind = (emp_data_AIS(i,:)+1)./2;
        ind = sum(basis.*v_ind) + 1;
        p_AIS_proba(ind) = p_AIS_proba(ind) + weight_AIS_list(i);
    end
    error_list_3(ind_s) = norm(prob_f_list - p_AIS_proba);

    prob_final_list = zeros(num_supp, 1);
    prob_final_list_c = zeros(num_supp, 1);
    prob_AIS_list = zeros(num_supp, 1);
    for s = 1:N_samples
        v1 = (emp_data(s,:)+1)./2;
        v2 = (emp_data_c(s,:)+1)./2;
        ind_1 = sum(basis.*v1) + 1;
        prob_final_list(ind_1) = prob_final_list(ind_1) + 1; 
        ind_2 = sum(basis.*v2) + 1;
        prob_final_list_c(ind_2) = prob_final_list_c(ind_2) + 1; 
    end    

    prob_re = prob_final_list./N_samples;
    prob_re_c = prob_final_list_c./N_samples;
    error_list_1(ind_s) = norm(prob_f_list - prob_re);
    error_list_2(ind_s) = norm(prob_f_list - prob_re_c);
end    

figure(1);
p11 = loglog(N_sample_list, error_list_1);
hold on
p12 = loglog(N_sample_list, error_list_2);
hold on
p13 = loglog(N_sample_list, error_list_3);
hold on
p16 = loglog(N_sample_list, error_list_6);
legend([p11 p12 p13 p16], 'Glauber + Genetic +BD','Glauber + BD',...
    'Glauber + Reweight','Monte Carlo', 'Location', 'Best');
xlabel('sample size');
ylabel('$\|\hat{p}-p^\ast\|_{2}$','interpreter','latex');

figure(2);
plot(1:1:num_supp,prob_f_list');
xlabel('Index');
ylabel('Probability');

final_data = zeros(1, N_sample_list(5));
for ks = 1:N_sample_list(5)
    vind = (emp_data(ks,:)+1)./2;
    final_data(ks) = sum(basis.*vind) + 1;
end    

figure(3);
histogram(final_data, 'Normalization', 'pdf');
xlabel('Index');
ylabel('Probability');