

clear; close all; clc

load('ex4data1.mat');

m = size(X);
Xmax = max(X(:))
Xmin = min(X(:))

sel = randperm(m);
sel = sel(1:100);
m = length(sel);


X = X(sel, :);



% On va maintenant créer la fonction displayData
colormap(gray);

% On définit les constantes du probleme
example_width = 20;
example_height = size(X, 2) / example_width;
pad = 1;
num_width = sqrt(m);
num_height = sqrt(m);

% La matrice qu'on enverra dans imagesc
mataff = zeros(pad + num_height * (example_height + pad), ...
                  pad + num_width * (example_width + pad));
size(mataff); % 211 * 211
mataff -= 1;


% On va pouvoir remplir mataff
current_ex = 1;
for i = 1:num_height
  for j = 1:num_width
    Xc = reshape(X(current_ex, :), example_height, example_width);
    Xc = Xc / max(abs(Xc(:)));
    nb_ligne = 2 + (i - 1) * (example_height + 1);
    nb_colonne = 2 + (j - 1) * (example_width + 1);
    mataff(nb_ligne:(nb_ligne + example_height - 1), ...
              nb_colonne:(nb_colonne + example_width - 1)) = Xc;
    current_ex += 1;
    if current_ex > m,
      break;
    end
  end
end

imagesc(mataff)
axis image off;
drawnow;

%ALRIGHT

      






