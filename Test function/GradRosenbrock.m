function gradR = GradRosenbrock(x)
    n = size(x, 1);
    gradR = zeros(n, 1);
    alpha = 100;
    
    % first component
    gradR(1) = 4.*alpha.*x(1).*(x(1).^2 - x(2)) + 2.*(x(1) - 1);
    % last component
    gradR(n) = 2.*alpha.*(x(n) - x(n-1).^2);
    
    % if the dimension is larger than 2
    if n>2
        gradR(2:n-1) = 2.*alpha.*( x(2:(n-1)) - x(1:(n-2)).^2 ) + ... 
                       4.*alpha.*x(2:(n-1)).*( x(2:(n-1)).^2 - x(3:n) ) +...
                       2.*( x(2:end-1) - 1 );
    end

end