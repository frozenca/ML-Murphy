loadData('sarcosData')
setSeed(0);
 
[Xtrain, mu, sigma] = standardizeCols(Xtrain);
Xtest = standardizeCols(Xtest, mu, sigma);
assert(approxeq(mean(Xtrain), zeros(1, size(Xtrain, 2))));
assert(approxeq(sqrt(var(Xtrain)), ones(1, size(Xtrain, 2))));
 
[ytrain, mu_ytrain] = centerCols(ytrain);
ytest = centerCols(ytest, mu_ytrain);
 
Ntrain = size(Xtrain, 1);
Ntest = size(Xtest, 1);
 sigma2 = var(ytrain, 1);
 
%% a. Standard linear regression
w = linregFitL2QR(Xtrain, ytrain(:, 1), 0);
ypred = Xtest * w;
SMSE_OLS = sum((ytest(:, 1) - ypred).^2)/(Ntest * sigma2(1))
 
%% b. K-means clustering - RBF network
perm = randperm(Ntrain);
K = 500;
prototypes = kmeansFit(Xtrain(perm(1:10000), :), K); 

sigma = 10;
Ktrain = kernelRbfSigma(Xtrain, prototypes', sigma);
Ktest = kernelRbfSigma(Xtest, prototypes', sigma);
 
Drbf = size(Ktrain, 2);
model = linregFitL2QR(Ktrain, ytrain(:,1), 0);
ypred = Ktest * model;
SMSE_rbf = sum((ytest(:, 1) - ypred).^2)/(Ntest * sigma2(1))

%% c. Multilayer perceptron
lambda = 1;
model = mlpRegressFitSchmidt(Xtrain, ytrain(:, 1), [30], lambda);
ypred = mlpRegressPredictSchmidt(model, Xtest);
SMSE_mlp = sum((ytest(:, 1) - ypred).^2)/(Ntest * sigma2(1))