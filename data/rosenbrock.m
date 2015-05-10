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

classdef rosenbrock
    properties
        a;
        sqrt_b;
    end
    
    methods
        function this = rosenbrock(a, b)
            if nargin < 2
                b = 100;
                if nargin < 1
                    a = 1;
                end
            end
            this.a = a;
            this.sqrt_b = sqrt(b);
        end
        
        function [r, J] = residuals(this, x)
            y = x(2,:);
            x = x(1,:);
            r = [x - this.a; this.sqrt_b * (y - x .* x)];
            if nargout > 1
                J = repmat([1 0; 0 this.sqrt_b], numel(x));
                J(2,1,:) = (-2 * this.sqrt_b) * shiftdim(x, -1);
            end
        end
        
        function c = cost(this, x)
            c = residuals(this, x);
            c = sum(c .* c)';
        end
        
        function h = render(this, num, range)
            if nargin < 3
                range = [-3 3 -3 3];
                if nargin < 2
                    num = 100;
                end
            end
            [y, x] = ndgrid(linspace(range(3), range(4), num(min(2, end))), linspace(range(1), range(2), num(1)));
            x = log1p(cost(this, [x(:)'; y(:)']));
            x = reshape(x / max(x), size(y));
            h = image(range(1:2), range(3:4), x, 'CDataMapping', 'scaled');
            colormap(parula(24));
        end
    end
end