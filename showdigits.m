%close all;
%clear all;

data = load('optdigits.tra');

trarget = data(:,end);
training = data(:,1:end-1);

for digit = training'
    imagesc(reshape(digit,8,8)');
    pause(0.25);
end

clear data;

