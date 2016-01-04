function [inProcessed] = preprocess(in)
%adds new features and normalizes each data image
%a scalling is added after normalisation

EXTRAS = 5;
inProcessed = zeros(size(in,1),size(in,2) + EXTRAS );
inProcessed(:,1:end-EXTRAS) = in;
% 
for i = 1:length(in)
    %duplicate points to give additional weight to the locations
    points_weighted = [];
    for j =  find(in(i,:))
        points_weighted= [points_weighted ones(1, in(i,j)) * j];
    end

    row = mod(points_weighted,8) + 1;
    col = floor(points_weighted/8) + 1;
  
    covariance = cov(row,col);
    inProcessed(i,end-EXTRAS+1:end) = [
        covariance(1,1) 
        covariance(2,2) 
        covariance(1,2) 
        mean(row)
        mean(col)
        ];
end
    
inProcessed = normr(inProcessed) * 6;

end