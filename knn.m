clear all;
close all;
load('data4.mat');

%% regulardata with knn
correct_rate = [];
for i=19
    Mdlknn = fitcknn(train_data,train_label,'NumNeighbors',i);
    classes = predict(Mdlknn,test_data);
    cp = classperf(test_label,classes);
    correct_rate = [correct_rate,cp.CorrectRate];
end

x = 19;
plot(x,correct_rate,'b-o');
correct_rate
% cp.CorrectRate = 0.6852

%% zscore data


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