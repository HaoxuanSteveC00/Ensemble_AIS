function y = oneD_GL_PDF(x, delta, beta, h)
dim = size(x,2);
Mat = diag((ones(dim,1)./h)) - diag((ones(dim-1,1)./h),-1);
Mvec = Mat*x';
Mvec = [Mvec; -(x(:,dim)')./h];
Evec1 = (delta./2).*(vecnorm(Mvec,2,1).^2);

Evec2 = (1./(4*delta)).*(sum((1-x.^2).^2, 2));

y = exp(-beta.*(Evec1' + Evec2));
end