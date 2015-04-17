%ROSENBROCK Compute the Rosenbrock function as two residuals
%
%   [r, J] = rosenbrock(x)
%
%IN:
% x - 2xN set of N 2D coordinates to evaluate the function at.
%
%OUT:
% r - 2xN set of residuals at the N input coordinates.
% J - 2x2xN set of residual Jacobians at the N input coordinates.

function [r, J] = rosenbrock(x)
a = 1;
sqrt_b = 10;
y = x(2,:);
x = x(1,:);
r = [x - a; sqrt_b * (y - x .* x)];
if nargout > 1
    J = repmat([1 0; 0 sqrt_b], numel(x));
    J(2,1,:) = (-2 * sqrt_b) * shiftdim(x, -1);
end
end