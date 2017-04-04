% original svm accuracy 0.5800
% z normalized 0.5825
% rbf function 0.5000

%{
Accuracy_all =

    0.5000    0.5000    0.5000    0.5000    0.5000    0.5000    0.5000    0.5000
    0.5000    0.5000    0.5000    0.5000    0.5000    0.5000    0.5000    0.5000
    0.5000    0.5000    0.5000    0.5000    0.5000    0.5000    0.5000    0.5000
    0.5000    0.5000    0.5000    0.5000    0.5000    0.5000    0.5000    0.5000
    0.5000    0.5000    0.5000    0.5000    0.5000    0.5000    0.5000    0.5000
    0.5000    0.5000    0.5000    0.5000    0.5000    0.5000    0.5000    0.5000
    0.5000    0.5000    0.5000    0.5000    0.5000    0.5000    0.5000    0.5000
    0.5000    0.5000    0.5000    0.5000    0.5000    0.5000    0.5000    0.5000


best_pairs =

    0.1000    0.0100


ans =

    0.5000


were all classified as 3

%}





clear all;
close all;
load('data3.mat');

%{
train_data = zscore(train_data);
test_data = zscore(test_data);
%}

%{
[COEFF,SCORE,latent,tsquare] = pca(train_data);
train_data = SCORE(:,1:4);

[COEFF,SCORE,latent,tsquare] = pca(test_data);
test_data = SCORE(:,1:4);
%}

%c = [0.1, 0.5, 1, 2, 5, 10, 20, 50,100,500];
sigma = [10,20,30,40,50,60,70,80,90,100];
%Accuracy_all = zeros(8);
accuracy = [];
%for j = 1:10
    for k = 1:10
        SVMmodel = fitcsvm(train_data,train_label,'KernelFunction','rbf','KernelScale',sigma(k));
        classes = predict(SVMmodel,test_data);
        cp = classperf(test_label,classes);
        %Accuracy_all(j,k) = cp.CorrectRate;
        accuracy = [accuracy,cp.CorrectRate];
    end
%end
accuracy

%Accuracy_all
%{
[num idx] = max(Accuracy_all(:));
[x y] = ind2sub(size(Accuracy_all),idx);
best_Accuracy = num;
best_pairs = [c(x),sigma(y)]


SVMmodel = fitcsvm(train_data,train_label,'KernelFunction','rbf','BoxConstraint',c(x),'KernelScale',sigma(y));
classes = predict(SVMmodel,test_data);
cp = classperf(test_label,classes);
cp.CorrectRate


%}
