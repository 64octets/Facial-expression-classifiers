%% regular MED classifier
% accuracy 0.5809
%%
clear all;
close all;
load('data2.mat');

train_a = train_data(1:3962,:);
train_b = train_data(3963:11153,:);
mu_a = mean(train_a);
mu_b = mean(train_b);
test_a = test_data(1:991,:);
test_b = test_data(992:2789,:);

wrong = 0;
for j = 1:991
    if ((test_a(j,:)-mu_a)*(test_a(j,:)-mu_a)') > ((test_a(j,:)-mu_b)*(test_a(j,:)-mu_b)')
        wrong = wrong + 1;
    end
end

for j = 1:1798
    if ((test_b(j,:)-mu_a)*(test_b(j,:)-mu_a)') < ((test_b(j,:)-mu_b)*(test_b(j,:)-mu_b)')
        wrong = wrong + 1;
    end
end

accuracy = (2789-wrong)/2789 


%{
%% if data zsocred acc = 0.5825
train_data = zscore(train_data);
test_data = zscore(test_data);
train_a = train_data(1:1000,:);
train_b = train_data(1001:2000,:);
mu_a = mean(train_a);
mu_b = mean(train_b);
test_a = test_data(1:200,:);
test_b = test_data(201:400,:);

wrong = 0;
for j = 1:200
    if ((test_a(j,:)-mu_a)*(test_a(j,:)-mu_a)') > ((test_a(j,:)-mu_b)*(test_a(j,:)-mu_b)')
        wrong = wrong + 1;
    end

    if ((test_b(j,:)-mu_a)*(test_b(j,:)-mu_a)') < ((test_b(j,:)-mu_b)*(test_b(j,:)-mu_b)')
        wrong = wrong + 1;
    end
end
(400-wrong)/400

%}
