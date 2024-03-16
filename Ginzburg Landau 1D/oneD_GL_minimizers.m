h = 0.01;
delta = 0.05;
beta = 1;
num = (1./h) - 1;
x_min_1 = rand(1, num);
x_min_2 = -rand(1,num);
eta = 0.0001;

% Find two minimizers via Gradient Descent
for i = 1:2000
    fprintf('%d\n', i);
    x_min_1 = x_min_1 + eta.*oneD_GL_grad(x_min_1, delta, beta, h);
    x_min_2 = x_min_2 + eta.*oneD_GL_grad(x_min_2, delta, beta, h);
end
x_min_1 = [0, x_min_1, 0];
x_min_2 = [0, x_min_2, 0];
x_axis = 0:h:1;

figure(1);
plot(x_axis, x_min_1);
xlabel('$r$','interpreter','latex');
ylabel('$x(r)$','interpreter','latex');
ylim([0 1.1]);

figure(2);
plot(x_axis, x_min_2);
xlabel('$r$','interpreter','latex');
ylabel('$x(r)$','interpreter','latex');
ylim([-1.1 0]);