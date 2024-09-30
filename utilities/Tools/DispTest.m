function ....
    [iterations, Failures, Time, minValue, gradNorm, argmin, MeanBtIterations, ...
    MeanBtFailures, MeanHessianModifications, GlobMin, ConvergenceRate ] = ... 
    DispTest(name_f, x0, trueMin, nameMethod,hGrad, hHess, typeGrad, kMax, c1, rho, btMax, gradTol, tolErr )

% The function executes tests in all possible configuration of the parameters given. 
% 
% INPUTS:
% x0 = column vector that represents the starting point.
% trueMin = column vector that represents the global minimizer.
% kMax = maximum number of iterations allowed to the method.
% c1, rho, btMax, gradTol = vectors containing the parameters for the
%                           tests.
% tolErr = tolerance on the relative error of xk.

switch name_f
    case 'Chained Rosenbrock'
        disp('CHAINED ROSENBROCK')
    otherwise
        disp('The typed function is not in the folder')
end

switch nameMethod
    case 'Finite differences'
        disp('MODIFIED NEWTON WITH FINITE DIFFERENCES')
    otherwise
        disp('CLASSIC MODIFIED NEWTON')
end

% length of the vectors
n = length(x0);
NgradTol = length(gradTol);
Nrho = length(rho);
Nc1 = length(c1);
NbtMax = length(btMax);

% OUTPUTS:
% let A be one of the outputs. A(i,j,k,l) is the result obtained 
% with this configuration:
% - tolerance on the gradient = gradTol(i);
% - rho = rho(j);
% - c1 = c1(k);
% - maximum number of backtracking iterations = btmax(l).
%
% iterations = Number of iterations achieved.
% Failures = contains if the algorithm has failed.
% Time = computational time 
% minValue = value of f at the "minimum" found
% gradNorm = value of the norm of the gradient at the minimum found
% argmin = minimum point found
% MeanBtIterations = mean number of backtracking iterations
% MeanBtFailures = mean number of backtracking failures
% MeanHessianModifications = mean number of modification to the hessian
% GlobaMin = global minimum found? GlobMin0(i,j,k,l) = 1 -> yes
% Convergence rate = Empirical convergence rate. ConvergenceRate(i,j,k,l) = 0 if there are
%                    too few iterations or the global minimum was not found.
 
iterations = zeros(NgradTol, Nrho, Nc1, NbtMax);
Failures = zeros(NgradTol, Nrho, Nc1, NbtMax);
Time = zeros(NgradTol, Nrho, Nc1, NbtMax);
minValue = zeros(NgradTol, Nrho, Nc1, NbtMax);
gradNorm = zeros(NgradTol, Nrho, Nc1, NbtMax);
argmin = zeros(NgradTol, Nrho, Nc1, NbtMax, n);
MeanBtIterations = zeros(NgradTol, Nrho, Nc1, NbtMax);
MeanBtFailures = zeros(NgradTol, Nrho, Nc1, NbtMax);
MeanHessianModifications = zeros(NgradTol, Nrho, Nc1, NbtMax);
GlobMin = ones(NgradTol, Nrho, Nc1, NbtMax);
ConvergenceRate = zeros(NgradTol, Nrho, Nc1, NbtMax);

disp('')
disp(['starting poit: ', num2str(x0')])
disp('')

% every possibile combination of gardTol, rho, c1, btMax is used to test
% the algorithm.
for i = 1:NgradTol
    for j = 1:Nrho
        for k = 1:Nc1
            for l = 1:NbtMax
                switch name_f
                    case 'Chained Rosenbrock'
                        switch nameMethod
                            case 'Finite differences'
                                tic
                                [xk, minValue(i,j,k,l), gradNorm(i,j,k,l),...
                                    iterations(i,j,k,l), xseq, ~, btseq, iseq, failBt,...
                                    Failures(i,j,k,l)] = ModifiedNewton_FinDiff...
                                    (@Rosenbrock, name_f, x0 ,hHess, hGrad, typeGrad, kMax, ...
                                    c1(k), rho(j), btMax(l), gradTol(i));
                                Time(i,j,k,l) = toc;
                            otherwise
                                tic
                                [xk, minValue(i,j,k,l), gradNorm(i,j,k,l), ...
                                    iterations(i,j,k,l), xseq, ~, btseq, iseq, failBt, ...
                                    Failures(i,j,k,l)] = ModifiedNewton ...
                                    (@Rosenbrock, x0, @GradRosenbrock, @HessianRosenbrock, ...
                                    kMax, c1(k), rho(j), btMax(l), gradTol(i));
                                Time(i,j,k,l) = toc;
                        end

                    case 'Chained Wood'
                        switch nameMethod
                            case 'Finite differences'
                                tic
                                [xk, minValue(i,j,k,l), gradNorm(i,j,k,l),...
                                    iterations(i,j,k,l), xseq, ~, btseq, iseq, failBt,...
                                    Failures(i,j,k,l)] = ModifiedNewton_FinDiff...
                                    (@ChainedWood, name_f, x0 ,hHess, hGrad, typeGrad, kMax, ...
                                    c1(k), rho(j), btMax(l), gradTol(i));
                                Time(i,j,k,l) = toc;
                            otherwise
                                tic
                                [xk, minValue(i,j,k,l), gradNorm(i,j,k,l), ...
                                    iterations(i,j,k,l), xseq, ~, btseq, iseq, failBt, ...
                                    Failures(i,j,k,l)] = ModifiedNewton ...
                                    (@ChainedWood, x0, @GradChainedWood, @HessianWood, ...
                                    kMax, c1(k), rho(j), btMax(l), gradTol(i));
                                Time(i,j,k,l) = toc;
                        end


                    case 'Chained Powel'
                        switch nameMethod
                            case 'Finite differences'
                                tic
                                [xk, minValue(i,j,k,l), gradNorm(i,j,k,l),...
                                    iterations(i,j,k,l), xseq, ~, btseq, iseq, failBt,...
                                    Failures(i,j,k,l)] = ModifiedNewton_FinDiff...
                                    (@ChainedPowel, name_f, x0 ,hHess, hGrad, typeGrad, kMax, ...
                                    c1(k), rho(j), btMax(l), gradTol(i));
                                Time(i,j,k,l) = toc;
                            otherwise
                                tic
                                [xk, minValue(i,j,k,l), gradNorm(i,j,k,l), ...
                                    iterations(i,j,k,l), xseq, ~, btseq, iseq, failBt, ...
                                    Failures(i,j,k,l)] = ModifiedNewton ...
                                    (@ChainedPowel, x0, @GradPowel, @HessianPowel, ...
                                    kMax, c1(k), rho(j), btMax(l), gradTol(i));
                                Time(i,j,k,l) = toc;
                        end                         
                end

                argmin(i,j,k,l,:) = xk;
                MeanBtIterations(i,j,k,l) = mean(btseq);
                MeanBtFailures(i,j,k,l) = mean(failBt);
                MeanHessianModifications(i,j,k,l) = mean(iseq);
                % if the relative error exceeds the tolerance -> xk is not
                % the golbal minimum.
                if norm(xk-trueMin)/norm(trueMin) > tolErr
                    GlobMin(i,j,k,l) = 0;
                
                elseif iterations(i,j,k,l) >= 3 && GlobMin(i,j,k,l) == 1
                    ConvergenceRate(i,j,k,l) = convergenceRateCalculator(xseq, trueMin);
                end
            end
        end
    end
end
disp('EMPIRICAL STATISTICS OF THE PARAMETER TUNING')
disp(['Number of trials/configurations: ', num2str(NbtMax*Nc1*NgradTol*Nrho)])
disp(['Number of algorithm failures(the algorithm has not found a local minimum): ', num2str(sum(Failures, "all"))])
disp(['Number of times the algorithm returns the global minimum point: ', num2str(sum(GlobMin, 'all'))])
disp('Computational time:')
disp(['- mean: ', num2str(mean(Time, "all")), ' s'])
disp(['- standard deviation: ', num2str(std(Time,1, 'all')), ' s '])
disp('Number of iterations:')
disp(['- minimum: ', num2str(min(iterations,[],'all'))])
disp(['- mean: ', num2str(mean(iterations, "all"))])
disp(['- max: ', num2str(max(iterations,[], "all"))])
disp('Empirical convergence rate:')
disp(['- minimum: ', num2str(min(ConvergenceRate,[], "all"))])
disp(['- mean: ', num2str(mean(ConvergenceRate, 'all','omitnan'))])
disp(['- max: ', num2str(max(ConvergenceRate, [], "all"))])
disp('Mean number of backtracking iterations (in each single configuration):')
disp(['- minimum: ', num2str(min(MeanBtIterations,[], "all"))])
disp(['- mean: ', num2str(mean(MeanBtIterations, "all"))])
disp(['- max: ', num2str(max(MeanBtIterations,[], "all"))])
disp('Mean number of backtracking failures (in each single configuration):')
disp(['- minimum: ', num2str(min(MeanBtFailures,[], "all"))])
disp(['- mean: ', num2str(mean(MeanBtFailures, "all"))])
disp(['- max: ', num2str(max(MeanBtFailures,[], "all"))])
disp('Mean number of modifications to the hessian (in each single configuration):')
disp(['- minimum: ', num2str(min(MeanHessianModifications,[],'all'))]) 
disp(['- mean: ', num2str(mean(MeanHessianModifications, "all"))])
disp(['- max: ', num2str(max(MeanHessianModifications,[],'all'))])
disp('-----------------------------------------------------------------------------------------------------------------')
disp('-----------------------------------------------------------------------------------------------------------------')             
end