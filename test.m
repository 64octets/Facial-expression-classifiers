35887 total
30000 training
5887 testing

angry 4953 = 3962+991
happy 8989 = 7191+1798
sad 6077 = 4861 +1216
% MED
% GED
% MAP
KNN 1 - 9
SVM
Decision Trees?
Random Forest?
Neural Networks




% replace label -1 with 0 so the cp could work out the accuracy
train_label( train_label==-1 )=0; 
c = [0.1, 0.5, 1, 2, 5, 10, 20, 50];
sigma = [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10];
Accuracy_all = zeros(8);

%% caculate accuracy for different parameters 
for j = 1:8
    for k = 1:8
            SVMmodel = fitcsvm(train_data,train_label);
            [predict_label,accuracy,prob_estimate] = predict(SVMmodel,test_data,train_data(test,:));

        Accuracy_all(j,k) = accuracy;
    end
end

%% find the best accuracy and its location
[num idx] = max(Accuracy_all(:));
[x y] = ind2sub(size(Accuracy_all),idx);
best_Accuracy = num;
best_pairs = [c(x),sigma(y)]

%% plot the ROC curves
SVMmodel = svmTrain(train_label(train,:),train_data(train,:),sprintf('-c %f -g %f', c(x), sigma(y)));
[predict_label,accuracy,prob_estimate] = svmpredict(train_label(test,:),train_data(test,:), SVMmodel);
[tpr,fpr,thresholds] = roc([train_label(test,:)].', prob_estimate');
figure;
plotroc([train_label(test,:)].', prob_estimate');


%%

sigma = [10,20,30,40,50,60,70,80,90,100
  Columns 1 through 9

    0.5425    0.5300    0.7150    0.6750    0.7025    0.6775    0.6725    0.6700    0.6650

  Columns 10 through 11

    0.6525    0.5975
%%

together = [feature;imgRow];
fea = together;
nsample = size(fea,1);
width = 48;
height = 48;
xy_coord = [0 0;1 1; 2 2;3 3;4 4;5 5;6 6;7 7;8 8;9 9;10 10;11 11];
digitsImages = reshape(fea', height, width, size(fea,1));
scale = 0.1;
skip = 1;
plotImages(digitsImages, xy_coord,scale,skip);
title('Digit 3 image after applying LLE');
xlabel('1st component'),ylabel('2nd component');


%%


figure
width = 87;
height = 86;
xy_coord = [0,0];
digitsImages = reshape(ImgVector, height, width, 1);
scale = 1;
skip = 1;
plotImages(digitsImages, xy_coord,scale,skip);
title('Digit 3 image after applying LLE');
xlabel('1st component'),ylabel('2nd component');



width = 48;
height = 48;
xy_coord = [0,0;0,0;0,0;0,0;0,0;0,0;0,0;0,0;0,0;0,0;0,0;];
digitsImages = reshape(feature', height, width, 11);
scale = 1;
skip = 1;
plotImages(digitsImages, xy_coord,scale,skip);
title('Digit 3 image after applying LLE');
xlabel('1st component'),ylabel('2nd component');

%train_label(train_label==3)=1;
%test_label(test_label==3)=1;
accuracy = [];

for i = 5:5
    net = patternnet(i);
    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio = 15/100;
    net.divideParam.testRatio = 15/100;
    net.trainParam.showWindow = false;
    net.trainParam.showCommandLine = false;
    [net,tr] = train(net,train_data',train_label');
    
    outputs = net(train_data');
    errors = gsubtract(train_label',outputs);
    performance = perform(net,train_label',outputs)
    
    % Test the Network
    %view(net)
    outputs = net(test_data');
    [c,cm,ind,per] = confusion(test_label',outputs);
    accuracy = [accuracy,1-c];
    
    outputs(outputs>1.5) = 3;
    outputs(outputs<=1.5) = 0;

    %figure
    %plotconfusion(test_label',pridictes);
    [c,cm,ind,per] = confusion(test_label',outputs);
    %precision = cm(1,1)/(cm(1,1)+cm(2,1));
    %recall = cm(1,1)/(cm(1,1)+cm(2,2));
    %F_mesure = 2 * precision * recall / (precision + recall);
    accuracy = [accuracy,1-c];
    
end

accuracy






% B =[1,2,3;4,5,6;7,8,9];
% B3 = reshape(B',1,[]);

fea = [feature;imgRow];
nsample = size(fea,1);
width = 48;
height = 48;
xy_coord = [0 0;1 1; 2 2;3 3;4 4;5 5;6 6;7 7;8 8;9 9;10 10;11 11;12 12];
digitsImages = reshape(fea', height, width, size(fea,1));
scale = 0.1;
skip = 1;
plotImages(digitsImages, xy_coord,scale,skip);
title('Digit 3 image after applying LLE');
xlabel('1st component'),ylabel('2nd component');




fea = [imgRow;imgRow];
nsample = size(fea,1);
width = 48;
height = 48;
xy_coord = [0 0;1 1];
digitsImages = reshape(fea', height, width, size(fea,1));
scale = 1;
skip = 1;
plotImages(digitsImages, xy_coord,scale,skip);
title('Digit 3 image after applying LLE');
xlabel('1st component'),ylabel('2nd component');




together = [feature];
fea = together;
nsample = size(fea,1);
width = 48;
height = 48;
xy_coord = [0 0;1 1; 2 2;3 3;4 4;5 5;6 6;7 7;8 8;9 9;10 10;11 11];
digitsImages = reshape(fea', height, width, size(fea,1));
scale = 0.1;
skip = 1;
plotImages(digitsImages, xy_coord,scale,skip);
title('Digit 3 image after applying LLE');
xlabel('1st component'),ylabel('2nd component');


figure
fea = [imgRow;imgRow];
nsample = size(fea,1);
width = 48;
height = 48;
xy_coord = [0 0;1 1];
digitsImages = reshape(fea', height, width, size(fea,1));
scale = 1;
skip = 1;
plotImages(digitsImages, xy_coord,scale,skip);
title('Digit 3 image after applying LLE');
xlabel('1st component'),ylabel('2nd component');
