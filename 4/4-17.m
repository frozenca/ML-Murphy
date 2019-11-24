clear all
rawdata = loadData('heightWeight');
data.Y = rawdata(:,1); % 1=male, 2=female
data.X = [rawdata(:,2) rawdata(:,3)]; % height, weight
maleNdx = find(data.Y == 1);
femaleNdx = find(data.Y == 2);
classNdx = {maleNdx, femaleNdx};

% Plot class conditional densities
for tied=[false true]
    for c=1:2
        X = data.X(classNdx{c},:);
        % fit Gaussian
        mu{c}= mean(X);
        if tied
            Sigma{c} = cov(data.X); % all classes
        else
            Sigma{c} = cov(X); % class-specific
        end
        % Plot data and model
        g{c} = gaussProb(data.X, mu{c}(:)', Sigma{c});
    end
    pred = (g{2} > g{1}) + 1;
    if tied
        error_LDA = 100 * (sum(pred ~= data.Y) / length(data.X));
    else
        error_QDA = 100 * (sum(pred ~= data.Y) / length(data.X));
    end
end
