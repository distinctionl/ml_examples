%COMPUTE_NORMALIZATION_FUNCTION
%
% [norm_func, X_norm] = compute_normalization_function(X)
%
% Compute a suitable linear normalization from training data.
%
%IN:
%   X - MxN set of M N-dimensional input vectors
%
%OUT:
%   norm_func - handle to function which linearly scales and offsets each
%               dimension of X independently, so as to give zero mean and
%               unit variance.
%   X_norm - MxN normalized input array, s.t. X_norm = norm_func(X).

function [norm_func, X] = compute_normalization_function(X)
m = mean(X, 1);
X = bsxfun(@minus, X, m);
s = 1 ./ (sqrt(mean(X .* X, 1)) + 1e-38);
norm_func = @(X) bsxfun(@times, bsxfun(@minus, X, m), s);
if nargout > 1
    X = bsxfun(@times, X, s);
end