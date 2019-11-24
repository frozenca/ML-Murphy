rawdata = dlmread('heightWeightData.txt');
male_data = rawdata(rawdata(:,1) == 1,2:3);
mu = mean(male_data);
Sigma = cov(male_data);

%% a
hold on
scatter(male_data(:,1), male_data(:,2));
gaussPlot2d(mu, Sigma);

%% b
stdev = sqrt([Sigma(1,1), Sigma(2,2)]);
std_male_data = (male_data - mu) ./ stdev;
std_mu = mean(std_male_data);
std_Sigma = cov(std_male_data);

figure;
hold on
scatter(std_male_data(:,1), std_male_data(:,2));
gaussPlot2d(std_mu, std_Sigma);

%% c
[V, D] = eig(Sigma);
W = (D^(-0.5))*(V');
white_male_data = W*((male_data - mu)');
white_mu = mean(white_male_data,2);
white_Sigma = white_male_data * white_male_data' * (1/length(male_data));
figure;
hold on;
scatter(white_male_data(1,:), white_male_data(2,:));
gaussPlot2d(white_mu, white_Sigma);