function [out] = softmax(in,w)
    [~,outSize]=size( w );
    x = in * w ;
    
    x = exp(x);
    out = x ./ repmat(sum(x')',1,outSize);
end