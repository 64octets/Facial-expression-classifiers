clear all;
close all;
load('data2.mat');

train_label(train_label==3)=1;
test_label(test_label==3)=1;

layers = [imageInputLayer([48 48 1])
          convolution2dLayer(5,20)
          reluLayer
          maxPooling2dLayer(2,'Stride',2)
          fullyConnectedLayer(10)
          softmaxLayer
          classificationLayer()];
      
options = trainingOptions('sgdm','MaxEpochs',15,'InitialLearnRate',0.0001);

convnet = trainNetwork(train_data,layers,options);




%{

%}