beta = 0.001;
alpha = 0.1;
x_axis = -13:0.001:13;
y_axis = beta.*(x_axis.^4) - alpha.*(x_axis.^2);
figure(1);
plot(x_axis, y_axis);
xlabel('$x$','interpreter','latex');
ylabel('$V(x)$','interpreter','latex');
xlim([-13 13]);