clc;
clear all;
load('data2.mat')

weight = ones(3,1);
step_size = 0.05;
number_of_iterations = 2500;
N = length(data(:,1));
ones_column = ones(200,1);
data = horzcat(ones_column,data);

summation = 0;
cost_function = zeros(number_of_iterations,1);
predicted_y = zeros(N,1);

%GD
for n = 1 : number_of_iterations
    
    summation = 0;
    for i1 = 1 : N
        summation = summation + ((-data(i1,4)+sigmoid(weight,data,i1))*(data(i1,1:3)'));
    end
    
    weight = weight - step_size * summation;
   
    
    for i2 = 1 : N
       y_hat = sigmoid(weight, data, i2);
       yi = data(i2,4);
       
       cost_function(n) = cost_function(n) + (yi-1)*log(1-y_hat)-yi*log(y_hat);
       
    end
end

weight;

for i3 = 1 : N
    predicted_y(i3) = sigmoid(weight, data, i3);
end

data = horzcat(data,predicted_y);

boundary = (@(x) -(weight(1)+weight(2)*x)/weight(3));

hold on
scatter(data(data(:,5)>0.5,2),data(data(:,5)>0.5,3),'b');
scatter(data(data(:,5)<0.5,2),data(data(:,5)<0.5,3),'r');
fplot(boundary);
axis([0 1 -0.3 0.4])
%fplot(@(x) (1+exp(-weight'*x)));
figure()
%hold on
%scatter(data(data(:,4)==1,2),data(data(:,4)==1,3),'b');
%scatter(data(data(:,4)==0,2),data(data(:,4)==0,3),'r');
plot(1:number_of_iterations, cost_function)
