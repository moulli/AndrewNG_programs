% On utilise ici l'�quation normal qui nous retourne direct
% la solution analytique
function theta = normalEquation(X, y)
  
  theta = pinv(X' * X) * X' * y;
  