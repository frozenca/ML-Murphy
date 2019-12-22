rawdata = load('spamData.mat');
Xtrain = rawdata.Xtrain;
ytrain = rawdata.ytrain;
Xtest = rawdata.Xtest;
ytest = rawdata.ytest;

% a
Xtrain_binarize = Xtrain > 0;
Xtest_binarize = Xtest > 0;
model = naiveBayesFit(Xtrain_binarize, ytrain);
yhat_train = naiveBayesPredict(model, Xtrain_binarize);
trainErr = mean(ytrain ~= yhat_train);
fprintf('Train error using naive Bayes with binarized data : %5.3f \n', trainErr);
yhat_test = naiveBayesPredict(model, Xtest_binarize);
testErr = mean(ytest ~= yhat_test);
fprintf('Test error using naive Bayes with binarized data : %5.3f \n', testErr);

% b
Xtrain_std = (Xtrain - mean(Xtrain(:))) ./ var(Xtrain(:));
model2 = discrimAnalysisFit(Xtrain_std, ytrain, 'diag');
yhat_train2 = discrimAnalysisPredict(model2, Xtrain_std);
Xtest_std = (Xtest - mean(Xtest(:))) ./ var(Xtest(:));
trainErr2 = mean(ytrain ~= yhat_train2);
fprintf('Train error using GDA with standardized data : %5.3f \n', trainErr2);
yhat_test2 = discrimAnalysisPredict(model2, Xtest_std);
testErr2 = mean(ytest ~= yhat_test2);
fprintf('Test error using GDA with standardized data : %5.3f \n', testErr2);