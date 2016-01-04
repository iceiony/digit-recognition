
correct = 0;
for p = 1:length(testTarg)
    %remove 1 since index starts at 1
    result = find(testOut{2}(p,:) == max(testOut{2}(p,:))) - 1;
    if (result ~= testTarg(p))        
        imagesc(reshape(digit,8,8)');
        pause(0.25);
    end
    
end
