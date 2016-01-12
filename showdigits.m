load('optimal_weights.mat');
w = optimalWeights;

test = load('optdigits.tes');
testIn = [ones(length(test),1) preprocess(test(:,1:end-1))];
testTarg = test(:,end);

testOut{1} = [ones(length(testIn),1) sigmoid(testIn,w{1})];
testOut{2} = softmax(testOut{1},w{2});

correct = 0;
for p = 1:length(testTarg)
    %remove 1 since index starts at 1
    result = find(testOut{2}(p,:) == max(testOut{2}(p,:))) - 1;
    if (result ~= testTarg(p))        
        
        imagesc(reshape(testIn(p,2:65),8,8)');
        pause(0.25);
    else 
        correct = correct+1;
    
    end 
end

disp(fprintf('Network Performance %0.2f%%',correct/length(test) * 100));