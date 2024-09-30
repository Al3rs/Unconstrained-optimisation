function[xk, fk, gradfkNorm, k, xseq, fseq, btseq, iseq, failBt, failNewton] = ...
    ModifiedNewton_FinDiff(f, name_f, x0,hHess, hGrad, typeGrad, kMax, c1, rho, btMax, gradTol)
%
    % [xk, fk, gradfkNorm, k, xseq, fseq, btseq, iseq, failBt, failNewton] = ...
    % ModifiedNewton_FinDiff(f, name_f, x0, h, typeGrad, kMax, c1, rho, btMax, gradTol)
    %
    % INPUT:
    % f: R^n -> R function to minimize.
    % name_f = name of the function. (it is needed to label graphs and 
    %          to simplify the computations of the Hessian).
    % x0 = column vector that represents the initial guess.
    % h = step used to calculate finite differences.
    % typeGrad = type of finite difference used to approximate the
    %            gradient.
    % kMax = maximum number of iteration allowed.
    % btMax = maximum number of iteration for backtracking.
    % c1 = factor for Armijo condition.
    % rho = reduction parameter for backtracking. 
    % gradTol = tolerance on the gradient. 
    %
    %
    % OUTPUT:
    % xk = minimizer found at iteration k.
    % fk = value of the function at iteration k.
    % k = iteration at which the method stops.
    % gradfkNorm = norm of the gradient evaluated at xk. 
    % xseq = matrix of the minimizers: xseq(:,k) is the minimizer found at
    %        iteration k.
    % btseq = vector of the backtracking iterations: btseq(k) is the number
    %         of iteration needed to satisfy Armijo condition at iteration
    %         k.
    % iseq = vector of the matrix modification: iseq(k) is the number of
    %        iterations needed to make the Hessian sufficiently positive
    %        definite at the iteration k.
    % failBt = vector of the backtracking failures: if failBt(k) = 1, iMax
    %          iterations are reached without satisfying Armijo condition.
    % failNewton = binary variable: if failNewton = 1, it means that the 
    %              algorithm reached the maximum number of iterations without 
    %              finding the minimum.
%

% FUNCTION HANDLES TO CHECK ARMIJO CONDITION
ArmijoCondition = @(alpha, x, fx, p, gradfx, c) f(x + alpha*p) - fx <= c*alpha*gradfx'*p;

% SAFETY CHECKS
% making sure not to fall into numerical errors
if hHess < eps
    hHess = eps;
end
if hGrad < eps
    hGrad = eps;
end
% making sure that x0 is a column vector
if size(x0,2) > 1
    x0 = x0';
end

% INIZIALIZATIONS
xk = x0;
fk = f(xk);
gradfk = findiff_grad(f, xk, hGrad, typeGrad);
gradfkNorm = norm(gradfk);
Hessfk = findiff_Hess(f, name_f, xk, hHess);
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
            R = chol(Bk);           % In order to be sure that Bk is sufficientlty positive definite we add 5*eps. 
            i = i-1;
        else
            SymPosDef = 1;
            tau = max(2*tau,beta/2);
            Bk = Hessfk + tau.*I;
        end
        
        i = i+1;
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
    
    % UPDATING f AND ITS DERIVATIVES
    fk = f(xk);
    fseq(:,k) = fk;  
    gradfk = findiff_grad(f, xk, hGrad, typeGrad);
    gradfkNorm = norm(gradfk);
    Hessfk = findiff_Hess(f, name_f, xk, hHess);

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


%%% FUNCTIONS THAT APROXIMATE DERIVATIVES

function [gradfx] = findiff_grad(f, x, h, type)
%
% function [gradf] = findiff_grad(f, x, h, type)
%
% Function that approximate the gradient of f in x (column vector) with the
% finite difference (forward/centered) method.
%
% INPUTS:
% f = function handle that describes a function R^n->R;
% x = n-dimensional column vector;
% h = the h used for the finite difference computation of gradf
% type = 'fw' or 'c' for choosing the forward/centered finite difference
% computation of the gradient.
%
% OUTPUTS:
% gradfx = column vector (same size of x) corresponding to the approximation
% of the gradient of f in x.


gradfx = zeros(size(x));

switch type
    case 'fw'
        fx = f(x);
        for l=1:length(x)
            gradfx(l) = (f([x(1:l-1); x(l)+h; x(l+1:end)]) - fx)/h;
        end
    case 'c'
        for l=1:length(x)
            gradfx(l) = (f([x(1:l-1); x(l)+h; x(l+1:end)]) - ...
                f([x(1:l-1); x(l)-h; x(l+1:end)]))/(2*h);
        end
    otherwise % repeat the 'fw' case
        fx = f(x);
        for l=1:length(x)
            gradfx(l) = (f([x(1:l-1); x(l)+h; x(l+1:end)]) - fx)/h;
        end
end

end


function [Hessfx] = findiff_Hess(f, name_f, x, h)
%
% [Hessf] = findiff_Hess(f, x, h)
%
% Function that approximate the Hessian of f in x (column vector) with the
% finite difference method.
%
% INPUTS:
% f = function handle that describes a function R^n->R;
% name_f = name of the function.
% x = n-dimensional column vector.
% h = the h used for the finite difference computation of Hessf
%
% OUTPUTS:
% Hessfx = n-by-n matrix corresponding to the approximation of the Hessian 
% of f in x.

n = length(x);
% inizializations
diagonalValues = zeros(n,1);
extraDiagonalValues = zeros(n-1,1);
fx = f(x);

switch name_f

    case 'Chained Rosenbrock'
        
        for j = 1:n
            % Elements on the diagonal
            diagonalValues(j) = (f([x(1:j-1); x(j)+h; x(j+1:end)]) - 2*fx + ...
                f([x(1:j-1); x(j)-h; x(j+1:end)]))/(h^2);
            
            % Extradiagonal elements
            if  j <= n-1
                %l = j+1;
                extraDiagonalValues(j) = (f([x(1:j-1); x(j) + h; x(j+1:j); x(j+1) + h; x(j+2:end)]) - ...
                    f([x(1:j); x(j+1)+h; x(j+2:end)]) - ...
                    f([x(1:j-1); x(j)+h; x(j+1:end)]) + fx)/(h^2);
            end
         end 
        
        values = [diagonalValues; ... % diagonal elements
             extraDiagonalValues; ...% upper-diagonal elements
             extraDiagonalValues]; % lower-diagonal elements
        
        rowIndeces = [1:n, ... % diagonal elements
                      1:(n-1), ... % upper-diagonal elements
                      2:n]; % lower-diagonal elements
        
        columnIndeces = [1:n, ... % diagonal elements
                    2:n, ... % upper-diagonal elements
                    1:(n-1)];% lower-diagonal elements
        
        % Hf(rowIndeces(i), columnIndeces(i)) = values(i)
        Hessfx = sparse(rowIndeces, columnIndeces, values);

    case 'Chained␣Wood'
             % Initialize diagonal values
             diagonalValues = zeros(n,1);
             % Initialize lower diagonal values
             lowerdiagonalValues = zeros(n-1,1);
             % Initialize second lower diagonal values
             lower2diagonalValues = zeros(n-2,1);
             % Caluclate the f(x) to speed up computations
             fx = f(x);
             for j=1:n
                 % Calculate f_plus to speed up computations
                 f_plus = f([x(1:j-1); x(j)+h; x(j+1:end)]);
                 % Elements on the diagonal
                 diagonalValues(j) = (f_plus- 2*fx + ...
                 f([x(1:j-1); x(j)-h; x(j+1:end)]))/(h^2);
                 if j == n
                    break
                 end
             % Extradiagonal elements
                if mod(j,2) == 1
                 l = j + 1;
                 lowerdiagonalValues(j) = (f([x(1:j-1); x(j) + h; x(j+1:l-1); x(l)
                 + h; x(l+1:end)])- ...
                 f([x(1:l-1); x(l)+h; x(l+1:end)])- ...
                 f_plus + fx)/(h^2);
                end
                 if mod(j,2) == 0 && j<n-1
                     l=j+2;
                     lower2diagonalValues(j) = (f([x(1:j-1); x(j) + h; x(j+1:l-1); x(l)
                     + h; x(l+1:end)])- ...
                     f([x(1:l-1); x(l)+h; x(l+1:end)])- ...
                     f_plus + fx)/(h^2);
                 end
             end
         % Hess_chained_wood(rowIndeces(i), columnIndeces(i)) = values(i)
         values = [diagonalValues; ... % diagonal elements
         lowerdiagonalValues; ...% upper-diagonal elements
         lowerdiagonalValues; ...% lower-diagonal elements
         lower2diagonalValues; ...% upper-second-diagonal elements
         lower2diagonalValues]; % lower-second-diagonal elements
         
         rowIndeces = [1:n, ... % diagonal elements
         1:(n-1), ... % upper-diagonal elements
         2:n, ... % lower-diagonal elements
         1:(n-2), ... % upper-second-diagonal elements
         3:n]; ... % lower-second-diagonal elements
         
         columnIndeces = [1:n, ... % diagonal elements
         2:n, ... % upper-diagonal elements
         1:(n-1), ...% lower-diagonal elements
         3:n, ... % upper-second-diagonal elements
         1:(n-2)]; % lower-second-diagonal elements
         
         Hessfx = sparse(rowIndeces, columnIndeces, values);
    

    case 'Chained Powel'
         % Initialize diagonal values
         diagonalValues = zeros(n,1);
         % Initialize lower diagonal values
         lowerdiagonalValues = zeros(n-1,1);
         % Initialize third lower diagonal values
         lower3diagonalValues = zeros(n-3,1);
         % Calculate f(x) 
         fx = f(x);
         for j=1:n
             % Calculate f_plus 
             f_plus = f([x(1:j-1); x(j)+h; x(j+1:end)]);
             % Elements on the diagonal
             diagonalValues(j) = (f_plus- 2*fx+ ...
             f([x(1:j-1); x(j)-h; x(j+1:end)]))/(h^2);
             if j == n
                break
             end
             % Extradiagonal elements
             l = j + 1;
             lowerdiagonalValues(j) = (f([x(1:j-1); x(j) + h; x(j+1:l-1); x(l) + h;
             x(l+1:end)])- ...
             f([x(1:l-1); x(l)+h; x(l+1:end)])- ...
             f_plus + fx)/(h^2);
            if mod(j,2) == 1 && j<n-2
                 l=j+3;
                 lower3diagonalValues(j) = (f([x(1:j-1); x(j) + h; x(j+1:l-1); x(l)
                 + h; x(l+1:end)])- ...
                 f([x(1:l-1); x(l)+h; x(l+1:end)])- ...
                 f_plus + fx)/(h^2);
            end
         end
         % Hess_chained_powel(rowIndeces(i), columnIndeces(i)) = values(i)
         values = [diagonalValues; ... % diagonal elements
         lowerdiagonalValues; ...% upper-diagonal elements
         lowerdiagonalValues; ...% lower-diagonal elements
         lower3diagonalValues; ...% upper-second-diagonal elements
         lower3diagonalValues]; % lower-second-diagonal elements
         
         rowIndeces = [1:n, ... % diagonal elements
         1:(n-1), ... % upper-diagonal elements
         2:n, ... % lower-diagonal elements
         1:(n-3), ... % upper-third-diagonal elements
         4:n]; ... % lower-third-diagonal elements
         
         columnIndeces = [1:n, ... % diagonal elements
         2:n, ... % upper-diagonal elements
         1:(n-1), ...% lower-diagonal elements
         4:n, ... % upper-third-diagonal elements
         1:(n-3)]; % lower-third-diagonal elements
         
         Hessfx = sparse(rowIndeces, columnIndeces, values);
    
         
    otherwise
        Hessfx = zeros(n,n);
        for j=1:n
            % Elements on the diagonal
            Hessfx(j,j) = (f([x(1:j-1); x(j)+h; x(j+1:end)]) - 2*f(x) + ...
                f([x(1:j-1); x(j)-h; x(j+1:end)]))/(h^2);
            % Extradiagonal elements
            for l=j+1:n
                Hessfx(l,j) = (f([x(1:j-1); x(j) + h; x(j+1:l-1); x(l) + h; x(l+1:end)]) - ...
                    f([x(1:l-1); x(l)+h; x(l+1:end)]) - ...
                    f([x(1:j-1); x(j)+h; x(j+1:end)]) + f(x))/(h^2);
                Hessfx(j,l) = Hessfx(l,j);
            end
        end
 end

end

