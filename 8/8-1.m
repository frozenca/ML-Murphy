rawdata = load('spamData.mat');
Xtrain = rawdata.Xtrain;
ytrain = rawdata.ytrain;
Xtest = rawdata.Xtest;
ytest = rawdata.ytest;

% a
Xtrain_std = (Xtrain - mean(Xtrain(:))) ./ var(Xtrain(:));
lambda = 9;
model_std = logregFit(Xtrain_std, ytrain, 'lambda', lambda);
Xtest_std = (Xtest - mean(Xtest(:))) ./ var(Xtest(:));
yhat_train = logregPredict(model_std, Xtrain_std);
trainErr_std = mean(ytrain ~= yhat_train);
fprintf('Train error with standardized features: %5.3f \n', trainErr_std);
yhat_test = logregPredict(model_std, Xtest_std);
testErr_std = mean(ytest ~= yhat_test);
fprintf('Test error with standardized features: %5.3f \n', testErr_std);

% b
Xtrain_transform = log(Xtrain + 0.1);
lambda = 5;
model_transform = logregFit(Xtrain_transform, ytrain, 'lambda', lambda);
Xtest_transform = log(Xtest + 0.1);
yhat2_train = logregPredict(model_transform, Xtrain_transform);
trainErr_transform = mean(ytrain ~= yhat2_train);
fprintf('Train error with transformed features: %5.3f \n', trainErr_transform);
yhat2_test = logregPredict(model_transform, Xtest_transform);
testErr_transform = mean(ytest ~= yhat2_test);
fprintf('Test error with transformed features: %5.3f \n', testErr_transform);

% c
Xtrain_binarize = Xtrain > 0;
lambda = 4;
model_binarize = logregFit(Xtrain_binarize, ytrain, 'lambda', lambda);
Xtest_binarize = Xtest > 0;
yhat3_train = logregPredict(model_binarize, Xtrain_binarize);
trainErr_binarize = mean(ytrain ~= yhat3_train);
fprintf('Train error with binarized features: %5.3f \n', trainErr_binarize);
yhat3_test = logregPredict(model_binarize, Xtest_binarize);
testErr_binarize = mean(ytest ~= yhat3_test);
fprintf('Test error with binarized features: %5.3f \n', testErr_binarize);