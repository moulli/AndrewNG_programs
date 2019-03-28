% On va ici d�finir un algorithme de r�gression lin�aire
% Le fichier notes.txt contient trois colonnes : 
%	- Le nombre de A's obtenu en premi�re ann�e
%	- Le temps de travail fourni en deuxi�me ann�e
%	- Le nombre de A's obtenu en deuxi�me ann�e


clear; close all; clc

% On commence par chopper les data et les incorporer dans nos variables de base
% Apr�s avoir v�rifi� qu'on �tait dans le bon directory
%cd = 'C:\Users\Hippolyte Moulle\Desktop\Machine Learning\TP tout seul'
data = load('notes.dat');
x = data(:, 1:2);
y = data(:, 3);
X = [ones(length(x), 1) x];
theta = zeros(3, 1); %On initialise le vecteur theta � 0

% On aimerait tracer nos data mais c'est en 3D

% On va appeler la fonction costFunction, qui va nous donner la fonction de co�t
J = costFunction(X, y, theta);
fprintf("Avec un theta initialise � 0,\nLa fonction de cout est %f\n", J)

% On initialise alpha et le nombre d'it�rations
alpha = 0.003;
nb_ite = 50000;


% On va maintenant pouvoir utiliser l'algorithme du gradient
[theta, listJ] = descendingGradient(X, y, theta, alpha, nb_ite);
fprintf("On obtient un theta de :\n");
fprintf('%f\n', theta);
fprintf("On va tracer l'evolution de J en fonction des iterations\n")
plot(listJ);
pause;



% Hop on passe � l'agorithme des valeurs normales
X = [ones(length(x), 1) x];
theta2 = normalEquation(X, y);
fprintf("Avec la methode de l'�quation normale, on obtient un theta de :\n")
fprintf("%f\n", theta2);

difftheta = abs(theta - theta2);
fprintf("On obtient une difference entre les deux thetas de :\n")
fprintf("%f\n", difftheta);

X * theta, X * theta2, y

costFunction(X, y, theta), costFunction(X, y, theta2)


