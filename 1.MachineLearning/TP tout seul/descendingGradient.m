% La fonction qui va nous rendre theta et une liste des J
function [theta, listJ] = descendingGradient(X, y, theta, alpha, nb_ite)
  
  listJ = zeros(1, nb_ite);
  m = length(y);
  
  for i = 1:nb_ite
    delta = X * theta;
    delta = delta - y;
    
    deltadiff = zeros(length(theta), 1);
    for j = 1:length(theta)
      delta = delta.*X(:, j);
      deltadiff(j) = sum(delta);
      deltadiff(j) = deltadiff(j) / m;
    end
    
    theta = theta - alpha*deltadiff;
    J = costFunction(X, y, theta);
    listJ(i) = J;
  end
end
   