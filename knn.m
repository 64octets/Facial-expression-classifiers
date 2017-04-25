clear all;
close all;
load('data2.mat');

%% regulardata with knn
correct_rate = [];
for i = 1:2:81
    Mdlknn = fitcknn(train_data,train_label,'NumNeighbors',i);
    classes = predict(Mdlknn,test_data);
    cp = classperf(test_label,classes);
    correct_rate = [correct_rate,cp.CorrectRate];
end

x = 1:2:81;
plot(x,correct_rate,'b-o');
xlabel('number of K');
ylabel('Accuracy');
title('Accuracy for various K');
legend('Correct rates');

correct_rate

%{
load('data_zscore.mat');

correct_rate = [];
for i=1:2:9
    Mdlknn = fitcknn(train_data,train_label,'NumNeighbors',i);
    classes = predict(Mdlknn,test_data);
    cp = classperf(test_label,classes);
    correct_rate = [correct_rate,cp.CorrectRate];
end

figure
x = 1:2:9;
plot(x,correct_rate,'b-o');
% cp.CorrectRate = 0.6852

%}