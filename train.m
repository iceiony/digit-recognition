clear all;
close all;

%load training data and pre process it for training
data = load('optdigits.tra');
data = addrotations(data);
in = data(:,1:end-1);
in = preprocess(in);

%load test data and preprocess it for later testing
test = load('optdigits.tes');
testIn = preprocess(test(:,1:end-1));
testTarg = test(:,end);

%because this is a classification transform targ into a binary output
targ = zeros(0,10);
for p = data(:,end)';
    targ(end+1,:) = zeros(1,10);
    targ(end,p + 1) = 1;
end

inSize = size(in,2);
outSize = size(targ,2);
trainSize = size(targ,1);

networkPerformance = zeros(10,1);
maxPerformance = 0;
optimalWeights = {};

%run multiple for optimisation success average
for runCount = 1:size(networkPerformance,1)
    
    regularisation = 0.001;
    miu = [0.1 0.1]; %learning rate
    batchSize = 20;
    hiddenSize = 60;
    
    w = {}; w_d={}; 
    w{1} = rand(inSize,hiddenSize) - 0.5;
    w{2} = rand(hiddenSize,outSize)- 0.5; %initialise the weight matrix for single layer
    out = {}; testOut = {};
    
    startIndex = 1 ;
    endIndex =  startIndex + batchSize;
    
%     entropy_cost = @(targ,out) -sum(sum(targ .* log(out)));
%     errors = []; %error cost function for plotting 

    tic
    for epoch = 1:28000

        in_batch = in(startIndex:endIndex,:);
        targ_batch = targ(startIndex:endIndex,:);

        out{1} = sigmoid(in_batch,w{1});
        out{2} = softmax(out{1},w{2});

        % collect entropy cost to test gradient is working
%         cost = entropy_cost(targ_batch,out{2});
%         errors(end+1)= cost;

        dif = targ_batch - out{2} ;

       
        w_d{1} = zeros(inSize,hiddenSize);
        w_d{2} = zeros(hiddenSize,outSize);
        for p=1:endIndex-startIndex
            w_d{2} = w_d{2} + out{1}(p,:)' * dif(p,:);
            w_d{1} = w_d{1} + in_batch(p,:)' * ( (w{2} * dif(p,:)')' .* out{1}(p,:) .* ( 1 - out{1}(p,:) )) ;
        end
        
        %update weights
        w{1} = w{1} + miu(1) * ( w_d{1} - regularisation * w{1} );
        w{2} = w{2} + miu(2) * ( w_d{2} - regularisation * w{2} ); 

        %update batch indexes
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

%     display gradient cost from training 
%     plot(1:length(errors),errors);
%     disp(fprintf('Final error:%d', errors(end)));

    %test performance of network
    testOut{1} = sigmoid(testIn,w{1});
    testOut{2} = softmax(testOut{1},w{2});

    correct = 0;
    for p = 1:length(testTarg)
        %remove 1 since index starts at 1
        result = find(testOut{2}(p,:) == max(testOut{2}(p,:))) - 1;
        correct = correct + ( result == testTarg(p));
    end

    networkPerformance(runCount) = correct/length(test) * 100;
    
    disp(fprintf('Prediction success: %0.2f%%',networkPerformance(runCount)));
    
    if networkPerformance(runCount) > maxPerformance
        optimalWeights = w;
        maxPerformance =  networkPerformance(runCount);
    end
end

disp(fprintf('Mean Performance: %0.2f%%\nVariance : %0.4f%%',mean(networkPerformance),var(networkPerformance)));

%save('optimal_weights.mat','optimalWeights','maxPerformance')
