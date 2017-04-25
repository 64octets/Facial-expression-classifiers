clear all;
close all;

%imageUint8 = imread('testimage2.png');  %unit8
%image1d = imageUint8(:,:,1);
%imageDouble = double(image1d); %double
%imshow(imageUint8)

feature = csvread('fer2013.csv',0,1);
label = csvread('fer2013.csv',0,0,[0,0,35886,0]);

angry_index = find(label==0);
angry_data = feature(angry_index,:);
angry_label = label(angry_index,:);
happy_index = find(label==3);
happy_data = feature(happy_index,:);
happy_label = label(happy_index,:);

train_data = [angry_data(1:3962,:);happy_data(1:7191,:)];
train_label = [angry_label(1:3962,:);happy_label(1:7191,:)];
test_data = [angry_data(3963:4953,:);happy_data(7192:8989,:)];
test_label = [angry_label(3963:4953,:);happy_label(7192:8989,:)];

train_label(train_label==3)=1;
test_label(test_label==3)=1;

train = [train_label,train_data];
test = [test_label,test_data];
% for python
csvwrite('train_11153.csv',train);
csvwrite('test_2789.csv',test);

% for matlab
save('data2.mat','train_data','train_label','test_data','test_label');

%{

train_data = [angry_data(1:1000,:);happy_data(1:1000,:)];
train_label = [angry_label(1:1000,:);happy_label(1:1000,:)];
test_data = [angry_data(1001:1200,:);happy_data(1001:1200,:)];
test_label = [angry_label(1001:1200,:);happy_label(1001:1200,:)];
%save('data3.mat','train_data','train_label','test_data','test_label');
train_label(train_label==3)=1;
test_label(test_label==3)=1;

train = [train_label,train_data];
test = [test_label,test_data];
csvwrite('train_csv.csv',train);
csvwrite('test_csv.csv',test);

%face1row = feature(4,:);
%face = reshape(face1row,[48,48]);
%man1 = uint8(face');
%figure
%imshow(man1);

%train_data = feature(1:30000,:);
%train_label = label(1:30000,:);
%test_data = feature(30001:35887,:);
%test_label = label(30001:35887,:);

train_data = [angry_data(1:3962,:);happy_data(1:7191,:)];
train_label = [angry_label(1:3962,:);happy_label(1:7191,:)];
test_data = [angry_data(3963:4953,:);happy_data(7192:8989,:)];
test_label = [angry_label(3963:4953,:);happy_label(7192:8989,:)];
save('data2.mat','train_data','train_label','test_data','test_label');

train_data = zscore([angry_data(1:3962,:);happy_data(1:7191,:)]);
train_label = zscore([angry_label(1:3962,:);happy_label(1:7191,:)]);
test_data = zscore([angry_data(3963:4953,:);happy_data(7192:8989,:)]);
test_label = zscore([angry_label(3963:4953,:);happy_label(7192:8989,:)]);
save('data_zscore.mat','train_data','train_label','test_data','test_label');
%}
