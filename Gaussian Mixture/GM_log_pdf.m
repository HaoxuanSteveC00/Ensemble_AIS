function y = GM_log_pdf(x, mus, sigmas, weights, gm)
num_size = size(x, 1);
y = zeros(num_size, 1);
num_w = size(weights, 2);
factor = repmat(pdf(gm,x),1,2);

for i = 1:num_w
    w_i = weights(i);
    mu_i = mus(i,:);
    sigma_i = sigmas((2*i-1):(2*i),:);
    f1 = repmat(mvnpdf(x, mus(i,:), sigma_i),1,2);
    f2 = (sigma_i\((x - repmat(mu_i, num_size, 1))'))';
    y = y + (w_i.*f1).*f2;
end    

y = y./factor;
end