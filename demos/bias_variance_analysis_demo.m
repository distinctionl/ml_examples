%BIAS_VARIANCE_ANALYSIS_DEMO
%
% This function demonstrates how to generate learning curves showing bias
% and variance, and how to analyze these curves to determine what changes
% to make to the learning proceedure, if any.

function scores = bias_variance_analysis_demo(X, y)
if nargin < 2
    % Load the data we are going to work with
    X = octane_data;
    rng(25259);
    X = X(randperm(size(X, 1)),:);
    y = X(:,end);
    X = X(:,1:end-1);
end

% Create all second-order features and add the offset feature to the data
m = numel(y);
n = size(X, 2);
X = [X reshape(bsxfun(@times, X, reshape(X, m, 1, n)), m, n*n)];

% Split into training, cross validation and test chunks, and normalize
m = m - ceil(0.4 * m);
[norm_func, X_train] = compute_normalization_function(X(1:m,:));
X_train = [ones(m, 1) X_train];
y_train = y(1:m);
X_cv = [ones(numel(y)-m, 1) norm_func(X(m+1:end,:))];
y_cv = y(m+1:end);
n = n + 1;

% Now compute the learning curves
M = unique(round(logspace(log10(10), log10(m), 15)));
scores = zeros(numel(M), 2, 4);
X = X_train;
y = y_train;
for a = numel(M):-1:1
    % Select the data subset to train on
    X = X(1:M(a),:);
    y = y(1:M(a));
    
    % Train the 4 models
    % 1: One feature only
    theta = linear_regression_train(X(:,1:2), y);
    scores(a,1,1) = linear_regression_cost(X(:,1:2), y, theta);
    scores(a,2,1) = linear_regression_cost(X_cv(:,1:2), y_cv, theta);
    
    % 2: Linear features only
    theta = linear_regression_train(X(:,1:n), y);
    scores(a,1,2) = linear_regression_cost(X(:,1:n), y, theta);
    scores(a,2,2) = linear_regression_cost(X_cv(:,1:n), y_cv, theta);
    
    % 3: Linear and quadratic features
    theta = linear_regression_train(X, y);
    scores(a,1,3) = linear_regression_cost(X, y, theta);
    scores(a,2,3) = linear_regression_cost(X_cv, y_cv, theta);
    
    % 4: Regularized linear and quadratic features
    theta = linear_regression_train(X, y, 5);
    scores(a,1,4) = linear_regression_cost(X, y, theta);
    scores(a,2,4) = linear_regression_cost(X_cv, y_cv, theta);
end

% Plot the graphs
T = {'One feature', sprintf('%d linear features', n-1), sprintf('%d linear, %d quadratic features', n-1, (n-1)^2), 'Linear & quadratic features, regularized'}; 
for a = 1:4
    figure(a);
    clf;
    set(gcf, 'Position', [100 100 350 250], 'Color', 'w');
    plot(M, scores(:,1,a));
    hold on
    plot(M, scores(:,2,a), 'r-');
    ylim([0 0.5]);
    legend('Training', 'Test');
    grid on
    xlabel 'Training set size';
    ylabel 'Average data cost';
    title(T{a});
end
