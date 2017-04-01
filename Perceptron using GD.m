clc; %clear console
clear all; %clear workspace
load('data3.mat')

weight = ones(3,1);
learning_rate = 0.08;
number_of_iterations = 200;

N = length(data(:,1)); %Length of my set of data
ones_column = ones(N,1); %These two lines put a column of ones in the data for the bias
data = horzcat(ones_column,data);
final_prediction = zeros(N,1); %Labels for last iteration
counter_error = zeros(number_of_iterations,1); % Vector with number of mistakes every iteration
prediction_every_iteration = zeros(N,1); %Labels calculated for every iteration

%SGD
for n = 1 : number_of_iterations
    summation = zeros(3,1);
    for i = 1 : N
        if data(i,4)*data(i,1:3)*weight <= 0 
            summation = summation + data(i,4)*data(i,1:3)';
        end
    weight = weight + ((learning_rate/N)*summation);
    end
    
    %For each iteration, we calculate all of our predicted y and calculate
    %and calculate error
    for i = 1 : N
        value = weight'* data(i,1:3)';
        prediction_every_iteration(i) =  classify(value);
    end
    
    for i = 1 : N
        if prediction_every_iteration(i) ~= data(i,4)
            counter_error(n) = counter_error(n) + 1;
        end
    end
    
    if counter_error(n) == 0
        n
        break
    end
    
    
end


for i = 1 : N
   y_predicted = weight'* data(i,1:3)';
   final_prediction(i) =  classify(y_predicted);
end
data = horzcat(data,final_prediction);

boundary = (@(x) -(weight(1)+weight(2)*x)/weight(3));

figure;

%subplot(2,1,1)
hold on
fplot(boundary);
scatter(data(data(:,5)==1,2),data(data(:,5)==1,3),'b');
scatter(data(data(:,5)==-1,2),data(data(:,5)==-1,3),'r');
axis([0 1 -0.3 0.4]);

figure()
%subplot(2,1,2)
plot(1:number_of_iterations, counter_error)

