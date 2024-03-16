d = 4;
dim = d^2;
num_supp = 2^(dim);
basis = 2.^(((dim-1):-1:0));

sub_mat = diag(ones(d-1,1),1) + diag(ones(d-1,1),-1);
sub_mat(1, d) = 1;
sub_mat(d, 1) = 1;
collection = repmat({sub_mat}, d, 1);
M1 = blkdiag(collection{:});
M2 = diag(ones(dim-d, 1), d) + diag(ones(dim-d, 1), -d); 
Mat_f = M1 + M2;
Mat_f(1:d, (dim-d+1):(dim)) = eye(d);
Mat_f((dim-d+1):(dim),1:d) = eye(d);
Mat_f = Mat_f./2;

N_samples = 512;
sample_set = linspace(1,N_samples,N_samples);
N_steps = 64; 
t_grid = linspace(0,1,N_steps+1);
delta_t = 1.0 / N_steps;
inner_iter = 1;
inner_iter_AIS = 3;
beta = 0.3; 

num_simu = 100;
error_list1 = zeros(num_simu,1);
error_list2 = zeros(num_simu,1);
error_list3 = zeros(num_simu,1);

E_f_list = zeros(num_supp, 1);
for l = 0:(num_supp-1)
    vec_l = dec2bin(l,dim)-'0';
    vec_l = 2*vec_l-1;
    E_f_list(l+1) = exp(-beta.*(vec_l*Mat_f*(vec_l')));
end
prob_f_list = E_f_list./(sum(E_f_list));
ref_prob = ones(num_supp,1);
ref_prob = ref_prob./sum(ref_prob);

for ind_n = 1:num_simu
    fprintf('Round %d\n', ind_n);
    emp_data = zeros(N_samples, dim);
    emp_BD_value = zeros(N_samples, 1);
    emp_BD_sum = 0;

    for i = 1:N_samples
        ind = randsample(1:1:num_supp,1,true,ref_prob) - 1;
        ind_vec = dec2bin(ind, dim) - '0';
        ind_vec = 2.*ind_vec -1;
        emp_data(i,:) = ind_vec;
        BD_rate = beta.*(ind_vec*Mat_f*ind_vec');
        emp_BD_value(i) = BD_rate;
        emp_BD_sum = emp_BD_sum + BD_rate;
    end 

    emp_data_c = emp_data;
    emp_data_AIS = emp_data;
    emp_BD_value_c = emp_BD_value;
    emp_BD_sum_c = emp_BD_sum;
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
                BD_rate_old = beta.*(pt*Mat_f*pt');
                BD_rate_new = beta.*(pt_flip*Mat_f*pt_flip');

                r1 = exp(-beta.*T.*(pt*Mat_f*(pt'))); 
                r2 = exp(-beta.*T.*(pt_flip*Mat_f*(pt_flip'))); 
                ratio = r2./(r1+r2);
                if rand < ratio
                    emp_data(j,:) = pt_flip;
                    % Update Birth Death Values
                    emp_BD_value(j) = BD_rate_new;
                    emp_BD_sum = emp_BD_sum + (BD_rate_new -...
                        BD_rate_old);
                end    

                % Genetic Algorithm
                rand_ind = randsample(candid, 1);
                pt = emp_data(j,:);
                pt_r = emp_data(rand_ind,:);
                BD_rate_old1 = beta.*(pt*Mat_f*pt');
                BD_rate_old2 = beta.*(pt_r*Mat_f*pt_r');
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
                BD_rate_new1 = beta.*(data_y1*Mat_f*data_y1');
                BD_rate_new2 = beta.*(data_y2*Mat_f*data_y2');
                r_y1 = exp(-T.*BD_rate_new1); 
                r_y2 = exp(-T.*BD_rate_new2); 
                rate = min(1,(r_y1*r_y2)./(r_pt*r_pt_r));

                if rand < rate
                    emp_data(j,:) = data_y1;
                    emp_data(rand_ind,:) = data_y2;
                    % Update Birth Death Values
                    emp_BD_value(j) = BD_rate_new1;
                    emp_BD_value(rand_ind) = BD_rate_new2;
                    emp_BD_sum = emp_BD_sum + (BD_rate_new1 -...
                        BD_rate_old1) + (BD_rate_new2 - BD_rate_old2);
                end    

                % Birth Death
                rate_j = emp_BD_value(j) - (emp_BD_sum./N_samples);
                pt = emp_data(j,:);
                BD_rate_old = beta.*(pt*Mat_f*pt');

                if rate_j > 0
                    prob = 1 - exp(-rate_j * delta_t);
                    if rand < prob
                        rand_index = randsample(candid, 1);
                        pt_s = emp_data(rand_index,:);
                        emp_data(j,:) = pt_s;
                        BD_rate_new = beta.*(pt_s*Mat_f*pt_s');
                        % Update Birth Death Values
                        emp_BD_value(j) = BD_rate_new;
                        emp_BD_sum = emp_BD_sum + (BD_rate_new -...
                            BD_rate_old);
                    end
                elseif rate_j < 0
                    prob = 1 - exp(rate_j * delta_t);
                    if rand < prob
                        rand_index = randsample(candid, 1);
                        pt_s = emp_data(rand_index,:);
                        emp_data(rand_index,:) = pt;
                        BD_rate_new = beta.*(pt_s*Mat_f*pt_s');
                        % Update Birth Death Values
                        emp_BD_value(rand_index) = BD_rate_old;
                        emp_BD_sum = emp_BD_sum + (BD_rate_old -...
                            BD_rate_new);
                    end
                end

            end
        end

        % Ensemble AIS without exploration
        for k = 1:inner_iter
            for j = 1:N_samples
                candid = sample_set(sample_set~=j);
                pt = emp_data(j,:);

                % Glauber Dynamics
                comp = unidrnd(dim);
                pt_flip = pt;
                pt_flip(comp) = -pt(comp);
                BD_rate_old = beta.*(pt*Mat_f*pt');
                BD_rate_new = beta.*(pt_flip*Mat_f*pt_flip');

                r1 = exp(-beta.*T.*(pt*Mat_f*(pt'))); 
                r2 = exp(-beta.*T.*(pt_flip*Mat_f*(pt_flip'))); 
                ratio = r2./(r1+r2);
                if rand < ratio
                    emp_data_c(j,:) = pt_flip;
                    % Update Birth Death Values
                    emp_BD_value_c(j) = BD_rate_new;
                    emp_BD_sum_c = emp_BD_sum_c + (BD_rate_new -...
                        BD_rate_old);
                end

                % Birth Death
                rate_j = emp_BD_value_c(j) - (emp_BD_sum_c./N_samples);
                pt = emp_data_c(j,:);
                BD_rate_old = beta.*(pt*Mat_f*pt');

                if rate_j > 0
                    prob = 1 - exp(-rate_j * delta_t);
                    if rand < prob
                        rand_index = randsample(candid, 1);
                        pt_s = emp_data_c(rand_index,:);
                        emp_data_c(j,:) = pt_s;
                        BD_rate_new = beta.*(pt_s*Mat_f*pt_s');
                        % Update Birth Death Values
                        emp_BD_value_c(j) = BD_rate_new;
                        emp_BD_sum_c = emp_BD_sum_c + (BD_rate_new -...
                            BD_rate_old);
                    end
                elseif rate_j < 0
                    prob = 1 - exp(rate_j * delta_t);
                    if rand < prob
                        rand_index = randsample(candid, 1);
                        pt_s = emp_data_c(rand_index,:);
                        emp_data_c(rand_index,:) = pt;
                        BD_rate_new = beta.*(pt_s*Mat_f*pt_s');
                        % Update Birth Death Values
                        emp_BD_value_c(rand_index) = BD_rate_old;
                        emp_BD_sum_c = emp_BD_sum_c + (BD_rate_old -...
                            BD_rate_new);
                    end
                end

            end
        end

        % Standard AIS with Glauber dynamics
        for j = 1:N_samples
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

    weight_AIS_list = exp(log_weight_list)./sum(exp(log_weight_list));
    p_AIS_proba = zeros(num_supp, 1);
    for i = 1:N_samples
        v_ind = (emp_data_AIS(i,:)+1)./2;
        ind = sum(basis.*v_ind) + 1;
        p_AIS_proba(ind) = p_AIS_proba(ind) + weight_AIS_list(i);
    end
    error_list3(ind_n) = norm(prob_f_list - p_AIS_proba);

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
    error_list1(ind_n) = norm(prob_f_list - prob_re);
    error_list2(ind_n) = norm(prob_f_list - prob_re_c);
end 

figure(1);
boxplot([error_list1, error_list2, error_list3],'notch', 'on',...
    'symbol', '*','colors','rbk');
legend(findobj(gca,'Tag','Box'),'Glauber + Reweight','Glauber + BD',...
    'Glauber + Genetic +BD');
ylabel('$\|\hat{p}-p^\ast\|_{2}$','interpreter','latex');