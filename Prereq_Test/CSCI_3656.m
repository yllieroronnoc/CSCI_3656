%{
    Prereq test
    Connor O'Reilly
    08/26/2021
%}

%% Q1
x = linspace(0,1,11)
%% Q2
y = x.^2 - 0.5.*x + 0.1625
%% Q3
figure(1)
plot(x,y)
xlabel('x')
ylabel('y')
title('x vs. y')
%% Q4
sum(y)
max(y)
sqrt(sum(y.^2))
%{
    Description:
        Q1: created vector containing 11 equally spaced points that cover
        the interval [0,1] using linspace(x,y,n). Linspace will generate n
        points covering x->y.
        Q2: using matlab matrix math, y was computed with each individual
        element in x.
        Q3: made a nice plot of x vs. y
        Q4: 
            a) used sum, which takes the sum of all array elements
            b) max(y) determines the maximum element in array y
            c) y.^2 , will compute the square of each element in y
            seperatly, sum(y.^2) will compute the sum of the array
            containing squared elements of y, sqrt(sum()) computes the 
            square root of the computed sum.
%}

%% Data
load('data.txt');
x = data(:,1)
y = data(:,2)
figure(2)
scatter(x,y)
% Graph has an upward trend as x increases past around 0.3 y also
% increases.