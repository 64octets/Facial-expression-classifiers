clear all;
close all;
load('data2.mat');

accuracy = [];
sigmas=[15,20,25,30];
sigmas2=[];
for i = sigmas
    SVMmodel = svmtrain(train_data,train_label,'kernel_function','rbf','rbf_sigma',i);
    classes = svmclassify(SVMmodel,test_data);
    cp = classperf(test_label,classes);
    cp.CorrectRate
    accuracy = [accuracy,cp.CorrectRate];
end

plot(sigmas,accuracy,'b-o');
xlabel('sigma value');
ylabel('Accuracy');
title('Accuracy for various sigma value');
legend('accuracy');

accuracy

%{

% best when sigma = 30 
%   0.5425    0.5300    0.7150    0.6750    0.7025    0.6775    0.6725
%   0.6700    0.6650    0.6525
c = [0.5, 1, 2, 5, 10];
sigma = [10,20,30,40,50,60,70,80,90,100];
    0.5050    0.6275    0.7000    0.5000    0.6775    0.6675    0.6475    0.6375    0.6325      0.6250
    0.5425    0.5300    0.7150    0.6750    0.7025    0.6775    0.6725    0.6700    0.6650      0.6525
    0.5425    0.5975    0.7150    0.6700    0.6900    0.6800    0.6750    0.6675    0.6625      0.6650
    0.5425    0.6975    0.5175    0.6775    0.6850    0.6925    0.6900    0.6800    0.5000      0.5000
    0.5425    0.6975    0.6950    0.5025    0.6575    0.6625    0.5000    0.5000    0.4975      0.5000
if use all images at sigma = 30
cp.CorrectRate = 0.7598
%}