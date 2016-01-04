function [out] = sigmoid(in,w)
    x = in * w ;
    out = 1 ./ (1+exp(-x));
end