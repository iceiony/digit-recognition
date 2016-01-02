function [out] = softmax(in,w)
    [~,outSize]=size( w );
    aux = exp(in * w);
    out = aux ./ repmat(sum(aux')',1,outSize);
end