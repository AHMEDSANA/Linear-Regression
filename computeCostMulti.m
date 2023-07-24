function J = computeCostMulti(X, y, theta)
m = length(y); 
J = 0;
for i=1:m;
J=J+1./(2*m)*(theta'*X(i,:)'-y(i)).^2;
end
end
