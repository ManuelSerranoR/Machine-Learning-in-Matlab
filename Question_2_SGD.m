clc;
clear all;
load('data2.mat')

weight = ones(3,1);
step_size = 0.05;
number_of_iterations = 40000;
N = length(data(:,1));
ones_column = ones(200,1);
data = horzcat(ones_column,data);

summation = 0;
cost_function = zeros(number_of_iterations,1);
predicted_y = zeros(N,1);

%SGD
for n = 1 : number_of_iterations
    
    aux = randi([1 200],1,1);
    summation = ((-data(aux,4)+sigmoid(weight,data,aux))*(data(aux,1:3)'));
    
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
figure()
plot(1:number_of_iterations, cost_function)