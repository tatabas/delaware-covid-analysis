%% Obtaining Data

%downloads the most up-to-date data
websave('us-states.csv','https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv');
%link to data: https://github.com/nytimes/covid-19-data/blob/master/us-states.csv

%loads that large data file
USstates = readtable("us-states.csv");

%specifically takes the delaware data
delaware = [USstates(USstates.state=="Delaware",:)];

%after getting desired portion, clearing large data file
clear USstates;clear ans;

%% Model for y = a*e^(kt) + b 

%% Estimates
%before bootstrapping the cumulative data we have to make the estimates as
%close to the expected value, so the function estimateCalc will be use to
%calculate the parameters for each window without bootstrapping

t = [0:13];
y = delaware.cases(:);
parameters = estimateCalc(t,y);

clear y;

%% Bootstrapping Fmincon

%tracking interval start days
day = 1;
%matrix for storing bootstrapped values with each column a set of
%bootstrapped values for a, then b, then k, then the next interval's set
bootfmin = [];
  %a matrix that lists every third number for a starting point in the
  %bootstrapped matrix
  e = [1:3:3*length(delaware.cases(:))];
  
totalIntervals = length(delaware.cases(:))-13;

while day <= totalIntervals
    t = [0:13];
    y = delaware.cases(day:day+13);
    %this following code will return a 1000X(totalIntervals*3) matrix, every
    %group of 3 columns is a set of parameters
    %parameters is a result of the function
    bootfmin(:,[e(day):e(day)+2]) = bootstrp(1000,@quickFun,t,y,parameters.a(day)*ones(14,1),parameters.k(day)*ones(14,1),parameters.b(day)*ones(14,1));
    day = day + 1;
end
clear c; clear k; clear e;

%saves the data
csvwrite('bootfmin_14.csv',bootfmin);


%% Storing A,K,B Parameters

%need to sort bootfmin to find the 95% conf interval
bootfmin_temp = sort(bootfmin);

%collecting all a, k, b into their own matrices
%collects every third entry starting from respective locations
temp_a = bootfmin_temp(:,[1:3:end]);
temp_k = bootfmin_temp(:,[2:3:end]);
temp_b = bootfmin_temp(:,[3:3:end]);

%starting variables to store means and upper/lower
mean_a=[];
mean_k=[];
mean_b=[];
upper_a=[];
upper_k=[];
upper_b=[];
lower_a=[];
lower_k=[];
lower_b=[];

for i = [1:totalIntervals]
    mean_a(i) = mean(temp_a(:,i)); %gets mean for each bootstrap sample
    mean_k(i) = mean(temp_k(:,i));
    mean_b(i) = mean(temp_b(:,i));
    upper_a(i) = temp_a(975,i); %gets upper for each bootstrap sample
    upper_k(i) = temp_k(975,i);
    upper_b(i) = temp_b(975,i);
    lower_a(i) = temp_a(25,i); %gets lower for each bootstrap sample
    lower_k(i) = temp_k(25,i);
    lower_b(i) = temp_b(25,i);
end


%storing the parameters in a table
bootedParameters = table(delaware.date(1:totalIntervals),mean_a',upper_a',lower_a',mean_k',upper_k',lower_k',mean_b',upper_b',lower_b');
bootedParameters.Properties.VariableNames = {'interval_start','mean_a','upper_a','lower_a','mean_k','upper_k','lower_k','mean_b','upper_b','lower_b'};

%clear unnecessary variables
clear i;clear temp_a;clear temp_k;clear temp_b;clear bootfmin_temp;clear mean_a;clear mean_k;clear mean_b;clear upper_a;clear upper_k;clear upper_b;clear lower_a; clear lower_b;clear lower_k;

%% Calculating Errors

%setting the matrix for storing errors
howFar = zeros(totalIntervals,1);
%incrementer
day = 1;
t = [0:13];

%loop to calculate the lst sqrs error for each window from the parameters
%that were obtained from the means of the bootstrapped data
for i = [1:totalIntervals]
    %the calculated parameters
    x = [bootedParameters.mean_a(day);bootedParameters.mean_k(day);bootedParameters.mean_b(day)];
    %cases values for this window of time
    y = delaware.cases(day:day+13);
    %calculating the lstSqrs error
    howFar(day) = lstSqrs(x,t,y);
    %increment to continue
    day = day+1;
end

%% Viewing Data

%close all

%plot viewing the k
figure
subplot(2,2,1)
plot(bootedParameters.interval_start(:),bootedParameters.mean_k(:),'linewidth',2);
title('K values in model y = a \times e^{kt} + b');
xlabel('14-day interval starting with');
ylabel('k value');
hold on

plot(bootedParameters.interval_start(:),bootedParameters.upper_k(:),':','linewidth',2);
plot(bootedParameters.interval_start(:),bootedParameters.lower_k(:),':','linewidth',2);

%plot viewing the a 
%figure
subplot(2,2,2)
plot(bootedParameters.interval_start(:),bootedParameters.mean_a(:),'linewidth',2);
title('A values y = a \times e^{kt} + b');
xlabel('14-day interval starting with');
ylabel('a value');
hold on

plot(bootedParameters.interval_start(:),bootedParameters.upper_a(:),':','linewidth',2);
plot(bootedParameters.interval_start(:),bootedParameters.lower_a(:),':','linewidth',2);

%plot viewing the b
%figure
subplot(2,2,3)
plot(bootedParameters.interval_start(:),bootedParameters.mean_b(:),'linewidth',2);
title('B values y = a \times e^{kt} + b');
xlabel('14-day interval starting with');
ylabel('b value');
hold on

plot(bootedParameters.interval_start(:),bootedParameters.upper_b(:),':','linewidth',2);
plot(bootedParameters.interval_start(:),bootedParameters.lower_b(:),':','linewidth',2);


%% Transition function between bootstrp and fmincon

function n = quickFun(t,y,a,k,b)
    %bootstrp passes a t vector that is vertical, we need horizontal
        %first one it passes is horizontal, then it passes vertical, so we
        %need to use if block here
    s = size(t);
    if s(1) > 1
        t = t';
    end
    %passing the variable required for it to be a vector of size 7, since
    %all values in the vector is the same, I am just taking the 1st value.
    a = a(1);
    k = k(1);
    b = b(1);
    n = fmincon(@(x)lstSqrs(x,t,y),[a;k;b],[],[],[],[],[-inf;-inf;-inf],[inf;20;inf]);
end

%% Objective Function Calculating Sum of Squared Error
% in model y = ae^(kt) + b

function fErr = lstSqrs(x,t,y)

%linear model y = ae^{kt} + b
%defining variables
a = x(1);
k = x(2);
b = x(3);

%calculating based on model
r = a*exp(k*t)+b;

%squared distance form is (x(2)-x(1))^2 + (y(2)-y(1))^2
%x component is nulled because x's are the same

%sum of squared errors
fErr = (y-r')'*(y-r');
end
