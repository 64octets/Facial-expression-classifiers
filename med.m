%% regular MED classifier
%error rate 0.59
clear all;
close all;
load('data2.mat');

train_a = train_data(1:3900,:);
train_b = train_data(5001:8900,:);
mu_a = mean(train_a);
mu_b = mean(train_b);
test_a = test_data(1:500,:);
test_b = test_data(1001:1500,:);

wrong = 0;
for j = 1:500
    if ((test_a(j,:)-mu_a)*(test_a(j,:)-mu_a)') > ((test_a(j,:)-mu_b)*(test_a(j,:)-mu_b)')
        wrong = wrong + 1;
    end

    if ((test_b(j,:)-mu_a)*(test_b(j,:)-mu_a)') < ((test_b(j,:)-mu_b)*(test_b(j,:)-mu_b)')
        wrong = wrong + 1;
    end
end
accuracy = (1000-wrong)/1000



load('data4.mat');

classes = [];
for j = 401:406
    if ((test_data(j,:)-mu_a)*(test_data(j,:)-mu_a)') > ((test_data(j,:)-mu_b)*(test_data(j,:)-mu_b)')
        classes = [classes,3];
    else
        classes = [classes,0];
    end
end
classes








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
