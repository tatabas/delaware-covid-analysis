%% Obtaining Data 

%downloads the most up-to-date data
websave('us-states.csv','https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv');
%link to data from New York Times Repository: https://github.com/nytimes/covid-19-data/blob/master/us-states.csv

%loads the total data file
USstates = readtable("us-states.csv");

%specifically takes delaware 
cumulativeDelaware = [USstates(USstates.state == "Delaware",:)];

%converts it into daily data
daily = zeros(length(cumulativeDelaware.cases(:)),1);
for i = [1:length(cumulativeDelaware.cases(:))]
    if i == 1
        daily(i) = cumulativeDelaware.cases(i);
    else
        daily(i) = cumulativeDelaware.cases(i)-cumulativeDelaware.cases(i-1);
    end
end

%takes the moving average because data is inconsistent
%first window 1:7, second window 2:8
dailyfil = zeros(length(daily)-7,1);
for i = [4:length(daily)-4]
    dailyfil(i-3)=mean(daily(i-3:i+3));
end

%turning data into a table to work with
delaware = table(cumulativeDelaware.date(3:length(dailyfil)+2),dailyfil);
delaware.Properties.VariableNames = {'date','cases'};

clear USstates; clear daily; clear dailyfil; clear cumulativeDelaware; clear i;

%% Model for y = a*e^(kt)

%% Calculating Data

%setting variabls
%a 13-day analysis window
t = [0:13]';
%begin counter
day = 1;
%the total windows that will be needed to ensure that all have 14 days
%totalIntervals = length(delaware.cases(:))-13; %for general analysis
totalIntervals = 117; %for data anlalyzed in document July 18, 2020
%setting size of the matrix
bootstat = zeros(1000,(2*totalIntervals));
%e stores the starting and ending points of the indices for data storage
e = [1:2:2*length(delaware.cases(:))];

%while loop to move through each 14 day window and calculate the model
%first window 1:14, second window 2:15
while day <= totalIntervals
    %specified set of data for this 14-day window
    y = log(delaware.cases(day:day+13));
    %storing the data
    bootstat(:,[e(day):e(day)+1])=bootstrp(1000,@regressAK,t,y);
    %incrementing to continue
    day = day + 1;
end

%ordering data
bootstat_sorted = sort(bootstat);
means = mean(bootstat_sorted);
%roughly calculating upper and lower bounds of 95% confidence interval
lower = bootstat_sorted(25,:);
upper = bootstat_sorted(975,:);

% inf error, refer to doc with proof
if any(bootstat == inf)
    disp('__________________');
    disp('re-run code, inf error');
    disp('__________________');
end  

%putting into a table to work with
params = table(delaware.date(1:size(bootstat,2)/2),means(1:2:end)',lower(1:2:end)',upper(1:2:end)',means(2:2:end)',lower(2:2:end)',upper(2:2:end)');
params.Properties.VariableNames = {'interval_start' 'means_a' 'lower_a' 'upper_a' 'means_k' 'lower_k' 'upper_k'};

clear bootstat; clear bootstat_sorted; clear means; clear lower; clear upper; clear e; clear day; clear t; clear y;

%% Calculating Lst Sq Errors
 
%calculating the least squares error for each parameter obtained from the
%14-day windows//calculating by comparing to the corresponding 
err1 = zeros(length(params.interval_start(:)),1);
%creating the numbers in the matrix for A
t = [0:13]';
for i = [1:length(params.interval_start(:))]
    %the y for each set of data
    y = log(delaware.cases(i:i+13));
    %the calculated a and k values
    x = [params.means_a(i);params.means_k(i)];
    A = [ones(14,1),t];
    err1(i) = (y-A*x)'*(y-A*x);
end

clear i; clear y; clear x; clear t; clear A;

%% Data Viewing

close all

figure
subplot(2,2,1)
plot(delaware.date(:),delaware.cases(:));
title({'Delaware';'Daily Cases in Delaware'});
xlabel('Dates');
ylabel('Cases');

subplot(2,2,2)
plot(params.interval_start(:),params.means_a,'linewidth',2);
hold on
title('A in y = a \times e^{kt}')
plot(params.interval_start(:),params.upper_a,':','linewidth',2);
plot(params.interval_start(:),params.lower_a,':','linewidth',2);
xlabel('14-Day Interval Start');

subplot(2,2,3)
plot(params.interval_start(:),params.means_k,'linewidth',2);
hold on
title('K in y = a \times e^{kt}')
plot(params.interval_start(:),params.upper_k,':','linewidth',2);
plot(params.interval_start(:),params.lower_k,':','linewidth',2);
xlabel('14-Day Interval Start');

subplot(2,2,4)
scatter(params.interval_start(:),err1);
title('Least Squares Errors');
xlabel('14-Day Interval Start');

%% Regression Function

function ps = regressAK(t,y)
    A = [ones(14,1),t];
    p = inv(A'*A)*A'*y;
    a = exp(p(1));
    k = p(2);
    ps = [a,k];
end
