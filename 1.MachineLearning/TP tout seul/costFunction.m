% Cette fonction va nous d�finir la fonction de co�t
function J = costFunction(X, y, theta)

m = length(y);

v = X * theta;
v = v - y;
v = v.^2;
J = (1/(2 * m))*sum(v);

end