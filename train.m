
%load training data 
data = load('optdigits.tra');
in = data(:,1:end-1);

%because this is a classification transform targ into a binary output
targ = zeros(0,10);
for i = data(:,end)';
    targ(end+1,:) = zeros(1,10);
    targ(end,i + 1) = 1;
end

inSize = length(in(1,:));
outSize = length(targ(1,:));
batchSize = 20;

w = rand(inSize,outSize); %initialise the weight matrix for single layer


out = softmax(in(1:batchSize,:),w);

