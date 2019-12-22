function main()

[xtrain, ytrain] = polyDataMake('sampling','thibaux');
N = length(xtrain);
wBatch = [ones(N,1) xtrain] \ ytrain;

ss = [];
for i=1:N
  [w(:,i),ss] = linregUpdateSS(ss, xtrain(i), ytrain(i));
end
figure;
hold on
h=plot(w(1,:), 'ko', 'linewidth', 2);
plot(w(2,:), 'r*', 'linewidth', 2);
plot(1:N, wBatch(1) * ones(1,N), 'k-', 'linewidth', 2);
plot(1:N, wBatch(2) * ones(1,N), 'r:', 'linewidth', 2);
h=legend('w0', 'w1', 'w0 batch', 'w1 batch');
axis_pct
title('Online linear regression')
ylabel('weights')
xlabel('time')

end

function [w,ss] = linregUpdateSS(ss, xnew, ynew)

if isempty(ss)
  ss.xbar = xnew;
  ss.ybar = ynew;
  ss.Cxx = 0;
  ss.Cxy = 0;
  ss.Cxx2 = 0;
  ss.Cxy2 = 0;
  ss.Cxx3 = 0;
  ss.Cxy3 = 0;
  ss.n = 1;
else
  ssOld = ss;
  ss.n = ss.n + 1;
  ss.xbar = ssOld.xbar + (1/ss.n)*(xnew - ssOld.xbar);
  ss.ybar = ssOld.ybar + (1/ss.n)*(ynew - ssOld.ybar);
  ss.Cxy = (1/ss.n)*(xnew * ynew + (ss.n - 1) * ssOld.Cxy + (ss.n - 1) * ssOld.xbar * ssOld.ybar - ss.n * ss.xbar * ss.ybar);
  ss.Cxx = (1/ss.n)*(xnew^2 + (ss.n - 1) * ssOld.Cxx + (ss.n - 1) * ssOld.xbar * ssOld.xbar - ss.n * ss.xbar * ss.xbar);
end

w1 = ss.Cxy / ss.Cxx;
w0 = ss.ybar - w1 * ss.xbar;
w = [w0; w1];

end
