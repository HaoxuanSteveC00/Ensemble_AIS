function y_grad = twoD_GL_grad(x, delta, beta, h)
dim = size(x,2);
dim0 = sqrt(dim);

M1 = (diag(ones(dim,1)) - diag(ones(dim-dim0,1),-dim0))./(h.^2);
M = (diag(ones(dim0,1)) - diag(ones(dim0-1,1),-1))./(h.^2);
collection = repmat({M}, dim0, 1);
M2 = blkdiag(collection{:});
Mat = M1 + M1' + M2 + M2';
Gvec1 = (delta./2).*(Mat*x')';

Gvec2 = (1./(delta)).*((x.^2-1).*x);
y_grad = -beta*(Gvec1 + Gvec2);
end