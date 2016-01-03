clear all;
close all;

%load training data and pre process it for training
data = load('optdigits.tra');
in = data(:,1:end-1);
in = preprocess(in);

%load test data and preprocess it for later testing
test = load('optdigits.tes');
testIn = preprocess(test(:,1:end-1));
testTarg = test(:,end);

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
% 
% mistakes = zeros(0,64);
% mistakeIndex = [];

for runCount = 1:10
    
    miu = 0.1; %learning rate
    batchSize = 30;
    startIndex = 1 ;
    endIndex =  startIndex + batchSize;

    w = rand(inSize,outSize); %initialise the weight matrix for single layer
    
    entropy_cost = @(targ,out) -sum(sum(targ .* log(out)));
%     errors = []; %error cost function for plotting 


    tic
    for epoch = 1:5000

        in_batch = in(startIndex:endIndex,:);
        targ_batch = targ(startIndex:endIndex,:);

        testOut = softmax(in_batch,w);

        % display entropy cost to test gradient is working
%         cost = entropy_cost(targ_batch,out);
%         errors(end+1)= cost;

        dif = targ_batch - testOut ;

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
            miu = miu / 1.1;
            startIndex = 1;
            endIndex = startIndex + batchSize;
        end

    end 
    toc

%     plot(1:length(errors),errors);
%     disp(fprintf('Final error:%d', errors(end)));

    %test performance of network
    testOut = softmax(testIn,w);

    correct = 0;
    for i = 1:length(testTarg)
        %remove one since index starts with 1
        result = find(testOut(i,:) == max(testOut(i,:))) - 1;
        correct = correct + ( result == testTarg(i));
%         if result ~= testTarg(i)
%             mistakes(end+1,:) = testIn(i,1:64);
%             mistakeIndex(end+1) = i;
%         end
    end

    networkPerformance(runCount) = correct/length(test) * 100;
    
    disp(fprintf('Prediction success: %0.2f%%',networkPerformance(runCount)));
    
    
end

disp(fprintf('Mean Performance: %0.2f%%\nVariance : %0.4f',mean(networkPerformance),var(networkPerformance)));
