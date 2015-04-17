%OPTIMIZATION_DEMO Demonstrate optimization of the Rosenbrock function
function optimization_demo(x)
if nargin < 1
    x = [-3; 3];
end
% Optimize
h = gauss_newton(@rosenbrock, x);
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
[Y, X] = ndgrid(linspace(range(3), range(4), n), linspace(range(1), range(2), n));
C = rosenbrock([X(:)'; Y(:)']);
C = reshape(sum(C .* C, 1), n, n);
clf;
image(range(1:2), range(3:4), log1p(C) ./ log1p(max(C(:))), 'CDataMapping', 'scaled');
colormap(parula(24));
hold on
plot(x(:,1), x(:,2), 'k-x');
end