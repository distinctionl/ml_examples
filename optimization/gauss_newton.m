%Gauss-Newton optimizer
classdef gauss_newton < handle
    properties (SetAccess = private)
        x;
        func;
    end
    methods
        function this = gauss_newton(residuals_fun, x0)
            this.x = x0;
            this.func = residuals_fun;
        end
        
        function [x, cost] = iterate(this)
            x = this.x;
            [r, J] = this.func(x);
            if size(J, 1) == numel(r)
                step = J \ r;
            else
                step = J';
                step = (step * J) \ (step * r);
            end
            x = x - step;
            this.x = x;
            if nargout > 1
                cost = r' * r;
            end
        end
    end
end