function y = many_well_PDF(x, alpha, beta)
dim_half = size(x, 2)/2;
x_first = x(:,1:dim_half);
x_second = x(:,(dim_half+1):(dim_half*2));

c1 = prod(exp(-beta.*x_first.^4+alpha.*(x_first.^2)),2);
c2 = prod(exp(-0.5.*((x_second).^2)), 2);
y = c1.*c2;
end