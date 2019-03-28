% La fonction qui va nous rendre theta et une liste des J
% On l'am�liore pour qu'elle s'arr�te elle m�me,
% quand la diff�rence des J est inf�rieure � une valeur,
% qu'on a d�finie
function [theta, listJ] = descendingGradient2(X, y, theta, alpha, evolution)
  
  m = length(y);
  
  J = costFunction(X, y, theta);
  listJ = [J];
  difference = evolution + 1;
  
  while difference > evolution
    delta = X * theta;
    delta = delta - y;
    
    deltadiff = zeros(length(theta), 1);
    for j = 1:length(theta)
      delta = delta.*X(:, j);
      deltadiff(j) = sum(delta);
      deltadiff(j) = deltadiff(j) / m;
    end
    
    theta = theta - alpha*deltadiff;
    J2 = costFunction(X, y, theta);
    difference = abs(J - J2);
    listJ = [listJ J2];
    J = J2;
  end
end