%LINEAR_REGRESSION_COST Cost and gradient of regularized linear regression
%
%   [J, grad] = linear_regression_cost(X, y, theta, lambda)

function [J, grad] = linear_regression_cost(X, y, theta, lambda)
m = numel(y); % number of training examples
X = [ones(m, 1) X];
hx = X * theta;
grad = hx - y;
J = (grad' * grad + (theta(2:end)' * theta(2:end)) * lambda) / (2 * m);
grad = sum(bsxfun(@times, grad, X), 1)' / m;
grad(2:end) = grad(2:end) + theta(2:end) * (lambda / m);
