clear all;
close all;
load('data2.mat');

train_label(train_label==3)=1;
test_label(test_label==3)=1;
accuracy = [];
layers = [1:20];

for i = layers
    net = patternnet(i);
    net.divideParam.trainRatio = 80/100;
    net.divideParam.valRatio = 20/100;
    [net,tr] = train(net,train_data',train_label');

    % Test the Network
    pridictes = net(test_data');
    pridictes(pridictes>0.5) = 1;
    pridictes(pridictes<=0.5) = 0;
    %figure, plotconfusion(test_gnd,pridictes);
    [c,cm,ind,per] = confusion(test_label',pridictes);
    accuracy = [accuracy,1-c];
    figure, plotconfusion(test_label',pridictes);

end
accuracy

plot(layers,accuracy,'b-o');
xlabel('layers');
ylabel('Accuracy');
title('Neural Network Classifier accuracy for various layers');
legend('accuracy');


