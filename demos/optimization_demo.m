%OPTIMIZATION_DEMO Demonstrate optimization of the Rosenbrock function
function optimization_demo(x, cost_func)
if nargin < 2
    cost_func = rosenbrock();
    if nargin < 1
        x = [-3; 3];
    end
end
% Optimize
h = gauss_newton(@(x) residuals(cost_func, x), x);
[x(:,end+1), s] = iterate(h);
for a = 1:100
    [x(:,end+1), s_] = iterate(h);
    if s_ == s
        break;
    end
    s = s_;
end
x = x';

% Compute the cost function over the range specified
n = 1000;
range = [min(x)-1; max(x)+2];
clf;
render(cost_func, n, range);
hold on
plot(x(:,1), x(:,2), 'k-x');
end