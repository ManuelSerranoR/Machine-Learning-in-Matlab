function [y] = sigmoid(weight, data, i)

    y = 1/(1+(exp(-(weight'*(data(i,1:3)')))));

end