clear; close all; clc

data = load('ex2data1.txt');
X = data(:, [1, 2]); y = data(:, 3);

plotData(X, y);

pause;


X = [ones(size(X, 1), 1) X X(:, 1).^(-1)];

theta = zeros(size(X, 2), 1);

[cost grad] = costFunction(theta, X, y);

options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, cost] = fminunc(@(t)(costFunction(t, X, y)), theta, options)


plotData(X(:,2:3), y);
hold on
plot_x = [floor(min(X(:,2))),  ceil(max(X(:,2)))];
abs_x = zeros(plot_x(2) - plot_x(1) + 1, 1);
for i = plot_x(1):plot_x(2)
  abs_x(i - plot_x(1) + 1) = i;
end
ord_y = (-1./theta(3)).*(theta(1) + theta(2).*abs_x + theta(4).*(abs_x.^(-1)));
plot(abs_x, ord_y)

