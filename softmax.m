function [out] = softmax(in,w)
    [~,outSize]=size( w );
    aux = in * w ;
    
%     aux = aux / 100;
    
    aux = exp(aux);
    out = aux ./ repmat(sum(aux')',1,outSize);
end