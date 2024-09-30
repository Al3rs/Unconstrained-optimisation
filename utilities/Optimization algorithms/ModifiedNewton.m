function[xk, fk, gradfkNorm, k, xseq, fseq, btseq, iseq, failBt, failNewton] = ...
    ModifiedNewton(f, x0, gradf, Hessf, kMax, c1, rho, btMax, gradTol)
%
    % [xk, fk, gradfkNorm, k, xseq, fseq, btseq, iseq, failBt, failNewton] = ...
    % ModifiedNewton(f, x0, gradf, Hessf, kMax, c1, rho, btMax, gradTol)
    %
    % INPUT:
    % f: R^n -> R function to minimize.
    % x0 = column vector that represents the initial guess.
    % gradf: R^n -> R^n gradient of the function. 
    % Hessf: R^n -> R^(n x n) Hessian matrix of f.
    % kMax = maximum number of iteration allowed.
    % c1 = factor for Armijo condition.
    % rho = reduction parameter for backtracking. 
    % btMax = maximum number of iteration for backtracking.
    % gradTol = tolerance on the gradient. 
    %
    %
    % OUTPUT:
    % xk = minimizer found at iteration k
    % fk = value function at iteration k
    % gradfkNorm = norm of the gradient evaluated at xk.
    % k = iteration at which the method stops
    % xseq = matrix of the minimizers: xseq(:,k) is the minimizer found at
    %        iteration k.
    % btseq = vector of the backtracking iterations: btseq(k) is the number
    %         of iteration needed to satisfy Armijo condition at iteration
    %         k.
    % failBt = vector of the backtracking failures: if failBt(k) = 1, iMax
    %          iterations are reached without satisfying Armijo condition.
    % failNewton = binary variable: if failNewton = 1, it means that the 
    %              algorithm reached the maximum number of iterations without 
    %              finding the minimum.
%

% FUNCTION HANDLE TO CHECK ARMIJO CONDITION
ArmijoCondition = @(alpha, x, fx, p, gradfx, c) f(x + alpha*p) - fx <= c*alpha*gradfx'*p;

% SAFETY CHECKS
% making sure that x0 is a column vector
if size(x0,2) > 1
    x0 = x0';
end

% INIZIALIZATIONS
xk = x0;
fk = f(xk);
gradfk = gradf(xk);
gradfkNorm = norm(gradfk);
Hessfk = Hessf(xk);
n = length(x0);
xseq = zeros(n, kMax);
fseq = zeros(1, kMax);
btseq = zeros(1, kMax);
failBt = zeros(1, kMax);
iseq = zeros(1,kMax);
failNewton = 0;
I = speye(n);
k = 0;

while k < kMax && gradfkNorm > gradTol

    k = k+1;

    % SOLVING Hf(xk)pk + gradf(xk) = 0 
    
    % making Hess(xk) a positive definite matrix
    beta = norm(Hessfk, 'fro');
    if all(diag(Hessfk) > 0)
        tau = 0;
    else
        tau = beta/2;

    end
    Bk = Hessfk + tau.*I;

    i = 0;
    SymPosDef = 1;
    while SymPosDef ~= 0 

        [R, SymPosDef] = chol(Bk);

        if SymPosDef == 0
            Bk = Bk + 5*eps*I;      % chol(Bk) returns cholesky factor even if Bk is SEMIpositive definite.
            R = chol(Bk);           % In order to be sure that Bk is positive definite we add 5*eps. 
        else
            SymPosDef = 1;
            tau = max(2*tau,beta/2);
            Bk = Hessfk + tau.*I;
        end
        if SymPosDef ~= 0
            tau = max(2*tau,beta/2);
            Bk = Hessfk + tau.*I;
            i = i+1;
        end
    end
    iseq(k) = i;

    % Descent direction
    pk = -R\(R'\gradfk);
    
    % SEARCHING A STEPLENGTH SATISFYING ARMIJO CONDITION
    alphak = 1;
    bt = 0;
    while ArmijoCondition(alphak, xk, fk, pk, gradfk, c1) == 0 && bt < btMax
        bt = bt+1;
        alphak = alphak*rho;
    end
    btseq(k) = bt;
    if bt == btMax &&  ArmijoCondition(alphak, xk, fk, pk, gradfk, c1) == 0
        failBt(k) = 1;
    end

    % UPDATING xk 
    xk = xk + alphak*pk;
    xseq(:,k) = xk;
    fk = f(xk);
    fseq(:,k) = fk;
    gradfk = gradf(xk);
    gradfkNorm = norm(gradfk);
    Hessfk = Hessf(xk);
end

if k == kMax && gradfkNorm > gradTol
    failNewton = 1;
end

% DROPPING UNUSEFUL DATA
xseq = xseq(:, 1:k);
fseq = fseq(:, 1:k);
btseq = btseq(:,1:k);
failBt = failBt(1:k);
iseq = iseq(1:k);

end