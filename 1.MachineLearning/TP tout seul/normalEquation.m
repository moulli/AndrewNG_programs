% On utilise ici l'équation normal qui nous retourne direct
% la solution analytique
function theta = normalEquation(X, y)
  
  theta = pinv(X' * X) * X' * y;
  