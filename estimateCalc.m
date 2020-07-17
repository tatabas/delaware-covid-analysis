function parameters = estimateCalc(t,cases)
%This function runs the modeling regression for each 14-day window
%   The parameters from this will be used as the estimates for wach window
%   in the bootstrapped function

%setting matrix for storing the parameters
p = [];
%incrementer
day = 1;
%total 14-day windows available in data
totalIntervals = length(cases)-13;

while day <= totalIntervals
    y = cases(day:day+13);
    if day == 1
        %the initial point is in form [a;k;b]. 1.9 and 0.3 come from the
        %first model which was based upon y = ae^kt. It serves as an
        %approporiate starting point for this analysis.
        p(day,:) = fmincon(@(x)lstSqrs(x,t,y),[1.9;0.3;0],[],[],[],[],[-inf;-inf;-inf],[inf;20;inf]);
        %a k and b can be negative due to policies that heed the growth of
        %the virus
    else
        %the parameters for the previous window make an approporate guess
        p(day,:) = fmincon(@(x)lstSqrs(x,t,y),[p(day-1,1);p(day-1,2);p(day-1,3)],[],[],[],[],[-inf;-inf;-inf],[inf;20;inf]);
    end
    %increment to continue
    day = day + 1;
end

% Storing A,K,B Parameters

%storing the parameters in a table date, a, k, b
parameters = table(p(:,1),p(:,2),p(:,3));
parameters.Properties.VariableNames = {'a', 'k', 'b'};

%the objective function for fmincon, minimzing the error means fitting the
%data

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

end

