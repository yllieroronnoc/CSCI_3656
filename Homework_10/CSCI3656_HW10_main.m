%% House keeping 
clear all; close all; clc;

%{
    CSCI 3656 HW10 
    Author: Connor O'Reilly
    Last Edited: 11/13/2021
    Email: coor1752@colorado.edu
%}

%% Part 1

% define function
f_p1 =@(x) sin(4.8 * pi * x);
%define functions first derivative found by hand
f1_truth =@(x) 4.8 * pi * cos(4.8 * pi * x);
%initilize values for h and pont to be evaluated
h = 2.^-(5:30); 
x_p1 = 0.2;

%compute relative error vs h for each three methods

%intialize storage and vars

truth = f1_truth(x_p1); %computed truth
len_h = length(h);
rel_err = zeros(len_h, 3);

for i = 1 : len_h
    
    %compute approximations
    fwd_approx = fwd_diff(f_p1 , x_p1, h(i));
    bck_approx = back_diff(f_p1 , x_p1, h(i));
    cnt_approx = cent_diff(f_p1 , x_p1, h(i));
    
    %compute relative error for the three methods
    rel_err(i,1) = abs(fwd_approx - truth) / abs( truth);
    rel_err(i,2) = abs(bck_approx - truth) / abs( truth);
    rel_err(i,3) = abs(cnt_approx - truth) / abs( truth);

end
% code for plot will be shown in plotting section

%find beginning of asymptotic range

% initialize arrays
log_h = log(h);
log_errf = log(rel_err(:,1)); %fwd diff
log_errb = log(rel_err(:,2)); %back diff
log_errc = log(rel_err(:,3)); %central diff

%find change points
%used
%https://www.mathworks.com/help/signal/ref/findchangepts.html#bu3nws1-ipt
%hopefully it works
ptsf = findchangepts(log_errf, 'MinThreshold', len_h);
ptsb = findchangepts(log_errb, 'MinThreshold', len_h);
ptsc = findchangepts(log_errc, 'MinThreshold', len_h);

%define regime for three methods
rngef = [ptsf(1) , ptsf(end)];
rngeb = [ptsb(1) , ptsb(end)];
rngec = [ptsc(1) , ptsc(end)];
%added to plots function

%find convergence rates
conv_f = ( log_errf(rngef(2)) - log_errf(rngef(1)) ) / ( log_h(rngef(2)) - log_h(rngef(1)) );
conv_b = ( log_errb(rngeb(2)) - log_errb(rngeb(1)) ) / ( log_h(rngeb(2)) - log_h(rngeb(1)) );
conv_c = ( log_errc(rngec(2)) - log_errc(rngec(1)) ) / ( log_h(rngec(2)) - log_h(rngec(1)) );


%% Part 2

%initialize method function
f1_p2 =@(x , h , f)  ( 1 /( 6*h ) )  * ( 2 * f(x+h) + 3 * f(x) - 6 * f(x-h) + f(x - 2*h) );
%intialize storage and vars
rel_errp2 = zeros(len_h,1);
for i = 1 : len_h
    %compute approximation
    p2_approx = f1_p2( x_p1 , h(i) , f_p1 );
    
    %compute relative error
    rel_errp2(i) = abs( p2_approx - truth) / abs( truth );
   
end

% code for plot will be shown in plotting section

%define asymptotic regime for finite difference approximation

%initialize 
log_errp2 = log(rel_errp2);
%find change points
%used
%https://www.mathworks.com/help/signal/ref/findchangepts.html#bu3nws1-ipt
%hopefully it works
ptsp2 = findchangepts(log_errp2, 'MinThreshold', len_h);
%define aymptotic regime for p2 plot
%looking at result
rngep2 = [ptsp2(1) , ptsp2(end-1)];

%plot in plotting section

%determine convergence rate 
conv_p2 = ( log_errp2(rngep2(2)) - log_errp2(rngep2(1)) ) / ( log_h(rngep2(2)) - log_h(rngep2(1)) );


%% Display
%part 1
fprintf('\n------------------------------------------------------------------\n')
fprintf('Part 1:')
fprintf('\n------------------------------------------------------------------\n\n\n')
fprintf('---------------------------------------------------------------------------------------------------\n')
fprintf('           Method             |    Estimated Convergence Rate    |    Theory Convergence Rate  ');
fprintf('|\n---------------------------------------------------------------------------------------------------\n')
fprintf(' Forward Difference     |                %0.5f                         |                       1   ', conv_f)
fprintf('                   |\n---------------------------------------------------------------------------------------------------\n')
fprintf(' Backward Difference  |                %0.5f                         |                       1   ', conv_b)
fprintf('                   |\n---------------------------------------------------------------------------------------------------\n')
fprintf(' Central Difference      |                %0.5f                         |                       2   ', conv_c)
fprintf('                   |\n---------------------------------------------------------------------------------------------------\n')

%part 2
fprintf('\n\n------------------------------------------------------------------\n')
fprintf('Part 2:')
fprintf('\n------------------------------------------------------------------\n\n\n')

fprintf('Using a numerical study similar to Problem one, the rate of convergence for this approximation is %0.5f\n', conv_p2)
%% Plotting
% part 1 plot 
%error vs h on a log-log scale for all approximations
figure(1)
loglog(h, rel_err(: , 1),'bo-')
hold on
loglog(h, rel_err(: , 2), 'k*-')
loglog(h, rel_err(: , 3) , 'm*-')
%add vertical lines for asymptotic range
%for foward diff
loglog([h(rngef(1)) h(rngef(1))], [10^-12 1], '--b', 'Linewidth',1.5)
loglog([h(rngef(2)) h(rngef(2))], [10^-12 1],'--b', 'Linewidth',1.5)
%backward diff
loglog([h(rngeb(1)) h(rngeb(1))], [10^-12 1], '--k', 'Linewidth',1.5)
loglog([h(rngeb(2)) h(rngeb(2))], [10^-12 1],'--k', 'Linewidth',1.5)
%central difference
loglog([h(rngec(1)) h(rngec(1))], [10^-12 1], '--m', 'Linewidth',1.5)
loglog([h(rngec(2)) h(rngec(2))], [10^-12 1],'--m', 'Linewidth',1.5)
grid on;
legend('Foward-Difference' , 'Backward-Difference', 'Central-Difference' ,'Asymptotic Regime Foward-Difference', '', 'Asymptotic Regime Backward-Difference','','Asymptotic Regime Central-Difference','', 'Location', 'northwest' )
title(' Relative error vs. h on log(x)-log(y) scale')
%from given code
ylabel('relative error: $|\,truth\,-\,approx\,|/|\,truth\,|$', 'interpreter', 'latex');
xlabel('$h$', 'interpreter', 'latex'); 
hold off;



%part 2
figure(2)
loglog(h, rel_errp2,'bo-')
hold on;
%plot asymptotic region
patch([h(rngep2(1)) h(rngep2(2)) h(rngep2(2)) h(rngep2(1)) ],[1 1 10^-14 10^-14],'magenta','FaceAlpha',.3);
grid on;
title(' Relative error vs. h on log(x)-log(y) scale')
%from given code
ylabel('relative error: $|\,truth\,-\,approx\,|/|\,truth\,|$', 'interpreter', 'latex');
xlabel('$h$', 'interpreter', 'latex'); 
legend('$f''(x) \approx \frac{1}{6h} [ 2f(x+h) + 3f(x) - 6f(x-h) + f(x-2h) ] $','Asymptotic Regime', 'Interpreter','latex','location','northwest')
hold off;



%% Functions

function [x1] = fwd_diff(f, x, h)
%{
    Purpose: matlab implementation of one-sided foward differerence to approximate
    first dereivative of f (f')
    
    Inputs: 
        f: function to derive
        x: evaluation point
        h: step size
    Output:
        x1: approximation of derivative at x
%}

    x1 = ( f(x+h) - f(x) ) / h;
    
end

function [x1] = back_diff(f, x, h)
%{
    Purpose: matlab implementation of one-sided backward differerence to approximate
    first dereivative of f (f')
    
    Inputs: 
        f: function to derive
        x: evaluation point
        h: step size
    Output:
        x1: approximation of derivative at x
%}

    x1 = ( f(x) - f(x-h) )/h;
    
end

function [x1] = cent_diff(f, x, h)
%{
    Purpose: matlab implementation of central differerence to approximate
    first dereivative of f (f')
    
    Inputs: 
        f: function to derive
        x: evaluation point
        h: step size
    Output:
        x1: approximation of derivative at x
%}

    x1 = ( f(x+h) - f(x-h) )/( 2 * h);
    
end

