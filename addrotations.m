function [rotated] = addrotations(data)

rotated = zeros(3*size(data,1),size(data,2));
for i = 1:size(data,1)
    j= (i-1)*3 + 1;
    
    rotated(j,:) = data(i,:);
  
    rotated(j+1,:) = zeros(1,65);
    rotated(j+2,:) = zeros(1,65);
    
    rotated(j+1,end) = data(i,end);
    rotated(j+2,end) = data(i,end);
    
    
    img = imrotate(reshape(data(i,1:64),8,8),-randi(20)+1,'nearest','crop');
    rotated(j+1,1:64) = img(:)';
    
    img = imrotate(reshape(data(i,1:64),8,8),randi(20)-1,'nearest','crop');
    rotated(j+2,1:64) = img(:)';
end


rotated = rotated(randperm(size(rotated,1)),:);

end
