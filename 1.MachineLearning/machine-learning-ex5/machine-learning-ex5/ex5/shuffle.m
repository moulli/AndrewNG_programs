function Xs = shuffle(X, lc = 0)
  
  %% Cette fonction va mélanger les lignes ou les colonnes d'une matrice donnée X
  %% Le paramètre lc permet de savoir si on mélange les lignes, les colonnes ou les deux
  %% Si lc = 1 : lignes, lc = 2 : colonnes, lc = 0 : les deux
  %% ATTENTION, les lignes et les colonnes restent en paquet,
  %% on ne melange pas les elements un par un
  
  [m, n] = size(X);
  
  if lc == 0 || lc ==1
    indices_m = zeros(m, 1);
    Xs = zeros(size(X));
    for i = 1:m
      x = 0;
      while all(indices_m - x) == 0
        x = ceil(m * rand());
      end
      indices_m(i) = x;
      Xs(i, :) = X(x, :);
    end
    X = Xs
  end
  
  if lc == 0 || lc ==2
    indices_n = zeros(n, 1);
    Xs = zeros(size(X));
    for i = 1:n
      x = 0;
      while all(indices_n - x) == 0
        x = ceil(n * rand());
      end
      indices_n(i) = x;
      Xs(:, i) = X(:, x);
    end
  end
  
end
