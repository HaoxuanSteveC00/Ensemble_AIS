function y = twoD_GL_PDF(x, delta, beta, h)
dim = size(x,2);
dim0 = sqrt(dim);

M1 = (diag(ones(dim,1)) - diag(ones(dim-dim0,1),-dim0))./h;
Mat = (diag(ones(dim0,1)) - diag(ones(dim0-1,1),-1))./h;
collection = repmat({Mat}, dim0, 1);
M2 = blkdiag(collection{:});

Mvec1 = M1*x';
Mvec1_T = (M1')*x';
Mvec2 = M2*x';
Mvec2_T = (M2')*x';

Evec1 = (delta./4).*(vecnorm(Mvec1,2,1).^2);
Evec1_T = (delta./4).*(vecnorm(Mvec1_T,2,1).^2);
Evec2 = (delta./4).*(vecnorm(Mvec2,2,1).^2);
Evec2_T = (delta./4).*(vecnorm(Mvec2_T,2,1).^2);
Evec3 = (1./(4*delta)).*(sum((1-x.^2).^2, 2));

y = exp(-beta.*(Evec1' + Evec1_T' + Evec2' + Evec2_T' + Evec3));
end