function alpha = convergenceRateCalculator(xseq, trueMin)
% convergence rate (alpha) obtained via Linear Regression:
%  log(||x_k+1 - x^*||) = alpha*log(||x_k - x^*||)

% exclude the last iteration where there could be log(0)
NumIterations = size(xseq, 2) - 1;

% errorNorm(i) = ||x_i - x^*|| (norm of the error at the iteration i)
errorNorm = sqrt(sum((xseq - trueMin).^2, 1))';

% vector of predictors
X = log(errorNorm(1:NumIterations-1));
% response 
y = log(errorNorm(2:NumIterations));

% convergence rate
alpha = X\y;

end