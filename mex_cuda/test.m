clear all;
order = 1;
N = 50;
M = 1000;
elim = 0.1;

[b,a] = butter(order,0.2);
u = ones(1,N);
z = filter(b,a,u);
e = elim*(2*rand(size(z))-1);
y=z+e;

theta = mex_cuda(u,y);
