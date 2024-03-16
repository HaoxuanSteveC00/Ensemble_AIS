function y_grad = oneD_GL_grad(x, delta, beta, h)
dim = size(x,2);
Mat = 2.*diag((ones(dim,1)./h)) - diag((ones(dim-1,1)./h),-1) -...
    diag((ones(dim-1,1)./h),1);
Gvec1 = (delta.*(Mat*x')')./h;
Gvec2 = (1./(delta)).*((x.^2-1).*x);
y_grad = -beta*(Gvec1 + Gvec2);
end