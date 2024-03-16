function y_grad = many_well_log_grad(x, alpha, beta)
num_samples = size(x, 1);
dim_half = size(x, 2)/2;
x_first = x(:,1:dim_half);
x_second = x(:,(dim_half+1):(dim_half*2));

y_grad = zeros(num_samples,dim_half*2);
y_grad(:,1:dim_half) = (4*beta).*(x_first.^3)-(2*alpha).*x_first;
y_grad(:,(dim_half+1):(dim_half*2)) = x_second;
end