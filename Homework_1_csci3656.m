%% Housekeeping
clc;
clear all;
close all;

%% Header

%{
    Homework 1 Code
    Author: Connor O'Reilly
    Last Edited  09/01/2021
%}

%% Q2
% Generate 8000 equally spaced points in the provided interval
int = linspace(1.92,2.08,8000);
%evaluate p1 and p2 (using horners for p2)
p1 = (int - 2).^9;
coeff_vec = [-512 , 2304 , -4608 , 5376 , -4032 , 2016 , -672 , 144 , -18 , 1];
p2 = Horners(coeff_vec , int);
%plotting
figure(1)
scatter(int, p1,'.');
hold on;
grid on;
xlabel('x')
ylabel( 'p1(x)')
title(' p1(x) at each point in the interval x = [1.92 , 2.08]')
figure(2)
scatter(int , p2 , '.');
hold on;
grid on;
xlabel('x')
ylabel( 'p2(x)')
title(' p2(x) at each point in the interval x = [1.92 , 2.08]')
%% Q3

%create values to be tested at
k = 0:12;
x_k = 10.^-k;
f1 = (1 - cos(x_k)) ./ (sin(x_k)).^2;
f2 = 1 ./ (1 + cos(x_k));
fmt=['x_k =' repmat(' %0.7f',1,numel(x_k))];
fprintf(fmt,x_k)
fprintf("\n")
fmt=['f1 =' repmat(' %0.7f',1,numel(f1))];
fprintf(fmt,f1)
fprintf("\n")
fmt=['f2 =' repmat(' %0.7f',1,numel(f2))];
fprintf(fmt,f2)
fprintf("\n")


function [eval] = Horners(coeff, x)
%{
Matlab implementation of Horner's algorithm
Inputs:
    coeff: constant coefficients
    x: values to evaluate, assumes x is a 1xn vector
Ouptuts:
    eval: evaluation of horners 
 %}
n = length(coeff);
eval = ones(1, length(x)) * coeff(n);
for i = n-1 : -1 : 1
    eval = x.* eval + coeff(i);
end
end