%% Obtaining Data

%downloads the most up-to-date data
websave('us-states.csv','https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv');
%link to data: https://github.com/nytimes/covid-19-data/blob/master/us-states.csv

%loads that large data file
USstates = readtable("us-states.csv");

%specifically takes the delaware data
delaware = [USstates(USstates.state=="Delaware",:)];

%after getting desired portion, clearing large data file
clear USstates;

%
%% Model for y = ce^kt
%

%% Bootstrapping the Regression

%create an exponential fit for data
    % using model y = c * e ^kt
    % made linear: ln(y) = ln(c) + kt 
    % y is the case number, t is the days passed since beginning
    % will be done with 14-day data analysis windows
    
%to store many values in a cell array
bootstat_k = [];
bootstat_c = [];
%to keep track of days
day = 1;
%totalIntervals = length(delaware.cases(:))-13;
totalIntervals = 117; %based on data analyzed in document for July 18

while day <= totalIntervals
    %current set of case values
    y = log(delaware.cases(day:day+13));
    %constant set of t
    t = [0:13]';
    %storing the bootstrapped values for use later
    bootstat_k(:,day) = bootstrp(1000,@regressK,t,y);
    bootstat_c(:,day) = bootstrp(1000,@regressC,t,y);
    day = day + 1;
end
%the product of this is a 1000XtotalIntervals array for K and C
%one for each 7-day interval


%
% Storing Bootstrapped Parameters
%


%this part of the code creates a table for both C and K that has the means,
%upper and lower bounds for each 7-day interval in a 67X3 table.

%sort the two arrays
bootstat_k_sorted = sort(bootstat_k);
bootstat_c_sorted = sort(bootstat_c);

%arrays to store the bootstrapped means, upper/lower bounds of 95% CI
Means_k = mean(bootstat_k_sorted);
%taking 25 and 975 as bounds because 95% CI has two 2.5% ends (25, 975)
Lower_k = bootstat_k_sorted(25,:);
Upper_k = bootstat_k_sorted(975,:);

%repeat for c
Means_c = mean(bootstat_c_sorted);
Lower_c = bootstat_c_sorted(25,:);
Upper_c = bootstat_c_sorted(975,:);

% since t(0) = c, if t(0) is 0 then log(0) is inf, and this messes with
% calculations
if any(Means_k == inf)
    disp('__________________');
    disp('re-run code, inf error');
    disp('__________________');
elseif any(Means_c == inf)
    disp('__________________');
    disp('re-run code, inf error');
    disp('__________________');
end  

%creating a table with the dates of incidences, means, lower/upper bounds
Kbootdata = table(delaware.date(1:totalIntervals), Means_k', Lower_k', Upper_k');
Cbootdata = table(delaware.date(1:totalIntervals),Means_c', Lower_c', Upper_c');
Kbootdata.Properties.VariableNames = {'interval_start', 'mean', 'lower', 'upper'};
Cbootdata.Properties.VariableNames = {'interval_start','mean', 'lower', 'upper'};

%clear variables that will not be used anymore
clear bootstat_k; clear bootstat_c; clear bootstat_k_sorted; clear bootstat_c_sorted; clear Means_k; clear Lower_k; clear Upper_k; clear Means_c; clear Lower_c; clear Upper_c; 

%% Some Data Viewing for y = ce^(kt) Model

close all

figure
scatter(Kbootdata.interval_start(:),Kbootdata.mean(:));
title('K Scatter Plot with error bounds in model y=ce^{kt}');
xlabel('7-day interval starting with');
ylabel('k value');
hold on

%this creates the lower bound of the CI as a magenta dots, error bounds
scatter(Kbootdata.interval_start(:),Kbootdata.lower(:),20,'m','filled');
% title('K Lower Scatter Plot');

%this creates the upper bound of the CI as a red dots, error bounds
scatter(Kbootdata.interval_start(:),Kbootdata.upper(:),20, 'r','filled');
% title('K Upper Scatter Plot');

%trends: downward trend for Mean, Upper, and Lower for K

figure
hold on
scatter(Cbootdata.interval_start,Cbootdata.mean(:));
title('C Scatter Plot in model y=ce^{kt}');
xlabel('7-day interval starting with');
ylabel('c value');

%this creates the lower bound of the CI as a magenta dots, error bounds
scatter(Cbootdata.interval_start(:),Cbootdata.lower(:),20,'m','filled');
% title('K Lower Scatter Plot');

%this creates the upper bound of the CI as a red dots, error bounds
scatter(Cbootdata.interval_start(:),Cbootdata.upper(:),20, 'r','filled');
% title('K Upper Scatter Plot');

%trends: upward slope for C means
    %becaus they are basically initial values 


%% Regression Functions used for Model y = ce^kt

function k = regressK(t,y)
    %create the A matrix for the bootstrapped sample
    A = [ones(14,1),t];
    %then multiplying in (A' A)^-1 A' y = p
    p = inv(A'*A)*A'*y;
    %solve for the constant
    k = p(2);
end

function c = regressC(t,y)
    %create the A matrix for the bootstrapped sample
    A = [ones(14,1),t];
    %then multiplying in (A' A)^-1 A' y = p
    p = inv(A'*A)*A'*y;
    %solve for the constant
    c = exp(p(1));
end
