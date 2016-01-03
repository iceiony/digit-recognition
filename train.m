clear all;
close all;

%load training data 
data = load('optdigits.tra');
in = data(:,1:end-1);

%because this is a classification transform targ into a binary output
targ = zeros(0,10);
for i = data(:,end)';
    targ(end+1,:) = zeros(1,10);
    targ(end,i + 1) = 1;
end

inSize = size(in,2);
outSize = size(targ,2);
trainSize = size(targ,1);

batchSize = 20;
startIndex = 1 ;
endIndex =  startIndex + batchSize;


w = rand(inSize,outSize); %initialise the weight matrix for single layer
entropy_cost = @(targ,out) -sum(sum(targ .* log(out)));

miu = 0.001; %learning rate
errors = []; %error cost function for plotting 

tic

for epoch = 1:2000
    
    in_batch = in(startIndex:endIndex,:);
    targ_batch = targ(startIndex:endIndex,:);

    out = softmax(in_batch,w);
    
    % display entropy cost to test gradient is working
    cost = entropy_cost(targ_batch,out);
    errors(end+1)= cost;
    
    dif = targ_batch - out ;

    w_d = zeros(inSize,outSize);
    for i=1:endIndex-startIndex
        w_d = w_d + in_batch(i,:)' * dif(i,:);
    end
    
    %update weights
    w = w + miu * w_d;
    
    %update batch indexs
    if endIndex ~= trainSize
        startIndex = endIndex;
        endIndex = min(trainSize,endIndex + batchSize);
    else
        startIndex = 1;
        endIndex = startIndex + batchSize;
    end
    
end 

toc

plot(1:length(errors),errors);
disp(fprintf('Final error:%d', errors(end)));

test = load('optdigits.tes');
out = softmax(test(:,1:end-1),w);
expected = test(:,end);

result = mod(find(round(out)==1),10);
correct = 0;
missed = zeros(0,10);
for i = 1:length(expected)
    %remove one since index starts with 1
    result = find(round(out(i,:))==1) - 1; 
    if isempty(result)
        result = find(out(i,:) == max(out(i,:))) - 1;
        correct = correct + ( result == expected(i));
        disp(fprintf('Prediction uncertain %d == %d ', result , expected(i) ));
    else
        correct = correct + ( result == expected(i));
    end
end

disp(fprintf('Prediction success: %0.2f%%',correct/length(test) * 100));

