% In this script the Modified Newton Method and the Modified Newton method
% with finite differences are tested on the Rosenbrock function.

clear all
close all
clc
%%
% ADD PATHS TO THE CURRENT DIRECTORY
addpath(fullfile(pwd, 'Test function/')); % Path to Rosenbrock functions
addpath(fullfile(pwd, 'utilities/Optimization algorithms/')); % Path to optimization algorithms
addpath(fullfile(pwd, 'utilities/Tools/')); % Path to tools

% FIXED PARAMETERS
kMax = 100; % maximum number of iterations
n = 10; % dimension
x0 = 1.2*ones(n,1); % "good" starting point
x1 = ones(n,1); % "bad" starting point
x1((1:2:n-1)) = -1.2;
trueMin = ones(n,1); % global minimum
hGrad = sqrt(eps); % parameter for the gradient calculation using finite differences
hHess = eps^(1/4); % parameter for the hessian calculation using finite differences
typeGrad = 'c'; % type of finite differences
tolErr = 0.2; % tollerance on the relative absolute error

% PARAMETERS TO TUNE
gradTol = 10.^(-4:-2:-12); 
rho = 0.1:0.2:0.9;
c1 = 10.^(-10:+2:-2);
btMax = 25:-5:5;
% length of the vectors
NgradTol = length(gradTol);
Nrho = length(rho);
Nc1 = length(c1);
NbtMax = length(btMax);

%% Classic modified newton starting from x0
[iterations0, Failures0, Time0, minValue0, gradNorm0, argmin0, MeanBtIterations0, ...
    MeanBtFailures0, MeanHessianModifications0, GlobMin0, ConvergenceRate0 ] = ... 
    DispTest('Chained Rosenbrock',x0, trueMin, 'Classic modified Newton',0,0,'c', kMax, c1, rho, btMax, gradTol, tolErr );

%% Modified newton with finite differences starting from x0
[iterations0FD, Failures0FD, Time0FD, minValue0FD, gradNorm0FD, argmin0FD, MeanBtIterations0FD, ...
    MeanBtFailures0FD, MeanHessianModifications0FD, GlobMin0FD, ConvergenceRate0FD ] = ... 
    DispTest('Chained Rosenbrock',x0, trueMin, 'Finite differences',hGrad,hHess, 'c', kMax, c1, rho, btMax, gradTol, tolErr );

%% Classic modified newton starting from x1
[iterations1, Failures1, Time1, minValue1, gradNorm1, argmin1, MeanBtIterations1, ...
    MeanBtFailures1, MeanHessianModifications1, GlobMin1, ConvergenceRate1 ] = ... 
    DispTest('Chained Rosenbrock',x1, trueMin, 'Classic modified Newton',0,0,'c', kMax, c1, rho, btMax, gradTol, tolErr );

%% Modified newton with finite differences starting from x1
[iterations1FD, Failures1FD, Time1FD, minValue1FD, gradNorm1FD, argmin1FD, MeanBtIterations1FD, ...
    MeanBtFailures1FD, MeanHessianModifications1FD, GlobMin1FD, ConvergenceRate1FD ] = ... 
    DispTest('Chained Rosenbrock',x1, trueMin, 'Finite differences',hGrad,hHess, 'c', kMax, c1, rho, btMax, gradTol, tolErr );