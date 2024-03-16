h = 0.025; 
delta = 0.125;
beta = 1;
num = (1./h) - 1;
x_min_1 = rand(1, num.^2);
x_min_2 = -rand(1, num.^2);
eta = 0.0001;

% Find two minimizers via Gradient Descent
for i = 1:5000
    fprintf('%d\n', i);
    x_min_1 = x_min_1 + eta.*twoD_GL_grad(x_min_1, delta, beta, h);
    x_min_2 = x_min_2 + eta.*twoD_GL_grad(x_min_2, delta, beta, h);
end

x_min_2D_1 = zeros(num + 2, num + 2);
x_min_2D_1(2:(num + 1), 2:(num + 1)) = (reshape(x_min_1, [num, num]))'; 
x_min_2D_2 = zeros(num + 2, num + 2);
x_min_2D_2(2:(num + 1), 2:(num + 1)) = (reshape(x_min_2, [num, num]))'; 
[X,Y] = meshgrid(0:h:1, 0:h:1);

figure(1);
mesh(X,Y,x_min_2D_1);
xlabel('x');
ylabel('y');

figure(2);
mesh(X,Y,x_min_2D_2);
xlabel('x');
ylabel('y');