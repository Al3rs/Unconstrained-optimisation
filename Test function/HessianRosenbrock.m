function Hf = HessianRosenbrock(x)
    n = size(x,1);
    alpha = 100;

    % diagonal elements
    diagonalValues = [12*alpha* x(1).^2 - 4*alpha*x(2) + 2; ... 
         12*alpha*x(2:(n-1)).^2 - 4*alpha*x(3:n) + 2 + 2*alpha; ...
         2*alpha];

    % lower diagonal elements and upper diagional elements
    extraDiagonalValues = -4*alpha.*x(1:(n-1));
 
    % Hf(rowIndeces(i), columnIndeces(i)) = values(i)
    values = [diagonalValues; ... % diagonal elements
             extraDiagonalValues; ...% upper-diagonal elements
             extraDiagonalValues]; % lower-diagonal elements
    rowIndeces = [1:n, ... % diagonal elements
                  1:(n-1), ... % upper-diagonal elements
                  2:n]; % lower-diagonal elements
    columnIndeces = [1:n, ... % diagonal elements
                2:n, ... % upper-diagonal elements
                1:(n-1)];% lower-diagonal elements

    Hf = sparse(rowIndeces, columnIndeces, values);

end