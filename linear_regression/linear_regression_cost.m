%LINEAR_REGRESSION_COST Cost and gradient of regularized linear regression
%
%   [J, grad] = linear_regression_cost(X, y, theta, lambda)

function [J, grad] = linear_regression_cost(X, y, theta, lambda)
if nargin < 4
    lambda = 0;
end
m = numel(y); % number of training examples
hx = X * theta;
grad = hx - y;
J = grad' * grad;
if lambda
    J = J + (theta(2:end)' * theta(2:end)) * lambda;
end
J = J / (2 * m);
if nargout > 1
    grad = sum(bsxfun(@times, grad, X), 1)' / m;
    if lambda
        grad(2:end) = grad(2:end) + theta(2:end) * (lambda / m);
    end
end
