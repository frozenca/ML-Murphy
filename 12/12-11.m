clear all

mu = [0 0 0];
sigma = [1 0 0; 0 1 0; 0 0 1];
rng = ('default')
z = mvnrnd(mu, sigma, 200);
x = zeros(200,3);

for i = 1 : 200
    x(i,1) = z(i,1);
    x(i,2) = z(i,1) + 0.001*z(i,2);
    x(i,3) = 10*z(i,3);
end

[lambda, psi] = factoran(x, 1);
coeff = pca(x);
    