function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, 
num_iters)
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
for iter = 1:num_iters;
z = zeros(size(X, 2),1);
for j=1:size(X,2);
 for i=1:m;
 z(j)=z(j)+1./m*(theta'*X(i,:)'-y(i)).*X(i,j)^min(j-1,1);
 end
end
for i=1:size(X,2);
theta(i)=theta(i)-alpha*z(i);
end
 
 J_history(iter) = computeCostMulti(X, y, theta);
end
end