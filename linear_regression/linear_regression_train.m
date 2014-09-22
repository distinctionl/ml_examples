%LINEAR_REGRESSION_TRAIN Train a linear regression model

function theta = linear_regression_train(X, y, lambda)
if nargin < 3
    lambda = 0;
end
if lambda
    % Minimize the regularized cost using a local optimizer
    theta = fmincg(@(theta) linear_regression_cost(X, y, theta, lambda), zeros(size(X, 2), 1), optimset('MaxIter', 200, 'GradObj', 'on'));
else
    % Solve the normal equation linear system
    theta = (X' * X) \ (X' * y);
end