%{
2000 image 

5 layer : 0.7783
5 layer : 0.6478 
10 layer : 0.7808
10 layer: 0.6552
1:20
0.6601    0.6355    0.6626    0.6478    0.6158    0.6601    0.6601    0.6601    0.6207  0.6379    0.6330

10000 image 

5:10
0.7214    0.7193    0.7167    0.7225    0.7185    0.7221

20:5:50
0.7257    0.7171    0.7228    0.7253    0.7264    0.7106    0.7167

1:5
0.7193    0.7311    0.7264    0.7300    0.6970

1:5
0.7225    0.7203    0.7103    0.7271    0.7175

%}



clear all;
close all;
load('data4.mat');

train_label(train_label==3)=1;
test_label(test_label==3)=1;
accuracy = [];
for i = 12
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
accuracy;
figure
x = 12;
plot(x,accuracy,'b-o');



%{
%train_label(train_label==3)=1;
%test_label(test_label==3)=1;

% Create a Pattern Recognition Network
net = patternnet(5);
%net.divideParam.trainRatio = 70/100;
%net.divideParam.valRatio = 15/100;
%net.divideParam.testRatio = 15/100;
% Train the Network
net = train(net,train_data',train_label');

% Test the Network
classes = net(test_data');
%classes(classes>1.5) = 3;
%classes(classes<=1.5) = 0;

%cp = classperf(test_label,classes');
%cp.CorrectRate

[c,cm,ind,per] = confusion(test_label',classes);
1-c




% View the Network
view(net)


%}
