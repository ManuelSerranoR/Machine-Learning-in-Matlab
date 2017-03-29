%Question 1
clc;
clear all;
load('data1.mat')
N = length(data(:,1));
feature = data(:,1);
label = data(:,2);

X_0 = ones(N,1);
X_1 = horzcat(X_0,feature);
X_2 = horzcat(X_1,feature.^2); % the dot before the square means element-wise operation
X_3 = horzcat(X_2,feature.^3); % the dot before the square means element-wise operation
X_4 = horzcat(X_3,feature.^4); % the dot before the square means element-wise operation
X_5 = horzcat(X_4,feature.^5);
X_6 = horzcat(X_5,feature.^6);
X_7 = horzcat(X_6,feature.^7);
X_8 = horzcat(X_7,feature.^8);
X_9 = horzcat(X_8,feature.^9);

WEIGHT_0 = (X_0'*X_0)\(X_0'*label);
WEIGHT_1 = (X_1'*X_1)\(X_1'*label);
WEIGHT_2 = (X_2'*X_2)\(X_2'*label);
WEIGHT_3 = (X_3'*X_3)\(X_3'*label);
WEIGHT_4 = (X_4'*X_4)\(X_4'*label);
WEIGHT_5 = (X_5'*X_5)\(X_5'*label);
WEIGHT_6 = (X_6'*X_6)\(X_6'*label);
WEIGHT_7 = (X_7'*X_7)\(X_7'*label);
WEIGHT_8 = (X_8'*X_8)\(X_8'*label);
WEIGHT_9 = (X_9'*X_9)\(X_9'*label);

hold on
scatter(data(:,1),data(:,2),2)
%figure()
fplot(@(x) WEIGHT_0);
fplot(@(x) WEIGHT_1(1)+WEIGHT_1(2)*x);
fplot(@(x) WEIGHT_2(1)+WEIGHT_2(2)*x+WEIGHT_2(3)*x^2);
fplot(@(x) WEIGHT_3(1)+WEIGHT_3(2)*x+WEIGHT_3(3)*x^2+WEIGHT_3(4)*x^3);
fplot(@(x) WEIGHT_4(1)+WEIGHT_4(2)*x+WEIGHT_4(3)*x^2+WEIGHT_4(4)*x^3+WEIGHT_4(5)*x^4);
fplot(@(x) WEIGHT_5(1)+WEIGHT_5(2)*x+WEIGHT_5(3)*x^2+WEIGHT_5(4)*x^3+WEIGHT_5(5)*x^4+WEIGHT_5(6)*x^5); 

