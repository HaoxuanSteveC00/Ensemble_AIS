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

N_samples = 65536;
sample_set = linspace(1,N_samples,N_samples);
N_steps = 64;
t_grid = linspace(0,1,N_steps+1);
delta_t = 1.0 / N_steps;
beta = 0.3;
ref_prob = ones(num_supp,1);
ref_prob = ref_prob./sum(ref_prob);

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

for i = 1:N_steps
    T = t_grid(i+1);
    fprintf('%d\n', T);

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
            emp_BD_sum = emp_BD_sum + (BD_rate_new - BD_rate_old);
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
            emp_BD_sum = emp_BD_sum + (BD_rate_new1 - BD_rate_old1) +...
                (BD_rate_new2 - BD_rate_old2);
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
                emp_BD_sum = emp_BD_sum + (BD_rate_new - BD_rate_old);
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
                emp_BD_sum = emp_BD_sum + (BD_rate_old - BD_rate_new);
            end
        end

    end    
end           

prob_final_list = zeros(num_supp, 1);
for ks = 1:N_samples
    vind = (emp_data(ks,:)+1)./2;
    indh = sum(basis.*vind) + 1;
    prob_final_list(indh) = prob_final_list(indh) + 1; 
end
prob_final_list = prob_final_list./sum(prob_final_list);

figure(1);
plot(1:1:num_supp,prob_final_list','red');
xlabel('Index');
ylabel('Probability');