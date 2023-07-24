data=diabetes2;
X = data(:, 1:8);
y = data(:, 9);
m = length(y);
[X mu sigma] = featureNormalize(X);
% Add intercept term to X
X = [ones(m, 1) X];
% Choose some alpha value
alpha = 0.01;
num_iters = 400;
%% Init Theta and Run Gradient Descent 
theta = zeros(9, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');

ylabel('Cost J');
%% Yes x, y Drawing
% Data points are represented by a cross symbol with a size of 10
plot(X, y, 'rx', 'MarkerSize', 6);
% Set up X axis
xlabel('population');
% Set up Y axis
ylabel('profit');
%% ============= visualization J(theta_0, theta_1) =============
fprintf('Visualizing J(theta_0, theta_1) ...\n')
% Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);
% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));
% Fill out J_vals
for i = 1:length(theta0_vals)
 for j = 1:length(theta1_vals)
 t = [theta0_vals(i); theta1_vals(j)];
 J_vals(i,j) = computeCostMulti(X(:,2:3),y, t);
 J_vals = J_vals';
 % Surface plot
figure;
 end
end
% Because of the way meshgrids work in the surf command, we need to
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';
% Surface plot
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');
% Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);

