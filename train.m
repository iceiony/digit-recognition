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

networkPerformance = zeros(10,1);

for runCount = 1:10
    batchSize = 20;
    startIndex = 1 ;
    endIndex =  startIndex + batchSize;

    w = rand(inSize,outSize); %initialise the weight matrix for single layer
    
    entropy_cost = @(targ,out) -sum(sum(targ .* log(out)));
%     errors = []; %error cost function for plotting 

    miu = 0.01; %learning rate

    tic
    for epoch = 1:3000

        in_batch = in(startIndex:endIndex,:);
        targ_batch = targ(startIndex:endIndex,:);

        out = softmax(in_batch,w);

        % display entropy cost to test gradient is working
%         cost = entropy_cost(targ_batch,out);
%         errors(end+1)= cost;

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
            miu = miu / 1.5;
            startIndex = 1;
            endIndex = startIndex + batchSize;
        end

    end 
    toc

%     plot(1:length(errors),errors);
%     disp(fprintf('Final error:%d', errors(end)));

    %test performance against test set
    test = load('optdigits.tes');
    out = softmax(test(:,1:end-1),w);
    expected = test(:,end);

    correct = 0;
    for i = 1:length(expected)
        %remove one since index starts with 1
        result = find(out(i,:) == max(out(i,:))) - 1;
        correct = correct + ( result == expected(i));
    end

    networkPerformance(runCount) = correct/length(test) * 100;
    
    disp(fprintf('Prediction success: %0.2f%%',networkPerformance(runCount)));
    
    
end
